#pragma once

#include <mkl_dfti.h>

#include <map>
#include <iostream>
#include <unordered_map>
#include <type_traits>
#include <mutex>
#include <zi/utility/singleton.hpp>

#include "../assert.hpp"
#include "../types.hpp"

namespace znn { namespace fwd {

#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)

#elif defined(ZNN_USE_DOUBLE_PRECISION)

#define ZNN_DFTI_TYPE DFTI_DOUBLE

#else

#define ZNN_DFTI_TYPE DFTI_SINGLE

#endif

typedef DFTI_DESCRIPTOR_HANDLE fft_plan;

class fft_transformer
{
private:
    vec3i rsize, csize;

    fft_plan f1, b1;
    fft_plan t2, t3;

    real scale;

public:
    ~fft_transformer()
    {
        DftiFreeDescriptor(&f1);
        DftiFreeDescriptor(&b1);
        DftiFreeDescriptor(&t2);
        DftiFreeDescriptor(&t3);
    }

    fft_transformer( vec3i const & _rsize,
                     vec3i const & _csize )
        : rsize(_rsize), csize(_csize)
    {
        scale = csize[0] * csize[1] * csize[2];

        STRONG_ASSERT(rsize[0]==csize[0]);
        STRONG_ASSERT(rsize[1]<=csize[1]);
        STRONG_ASSERT(rsize[2]<=csize[2]);

        csize[0] /= 2; csize[0] += 1;

        MKL_LONG status;

        // Out-of-place
        // Real to complex / complex to real along x direction
        // Repeated along z direction
        // Will need filter.y calls for each y
        {
            MKL_LONG strides_in[2]  = { 0, rsize[1] * rsize[2] };
            MKL_LONG strides_out[2] = { 0, csize[1] * csize[2] };

            status = DftiCreateDescriptor( &f1, ZNN_DFTI_TYPE,
                                           DFTI_REAL, 1, rsize[0] );
            status = DftiCreateDescriptor( &b1, ZNN_DFTI_TYPE,
                                           DFTI_REAL, 1, rsize[0] );

            status = DftiSetValue( f1 , DFTI_CONJUGATE_EVEN_STORAGE,
                                   DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( b1 , DFTI_CONJUGATE_EVEN_STORAGE,
                                   DFTI_COMPLEX_COMPLEX );

            status = DftiSetValue( f1, DFTI_PLACEMENT, DFTI_NOT_INPLACE );
            status = DftiSetValue( b1, DFTI_PLACEMENT, DFTI_NOT_INPLACE );

            status = DftiSetValue( f1, DFTI_INPUT_STRIDES, strides_in );
            status = DftiSetValue( f1, DFTI_OUTPUT_STRIDES, strides_out );

            status = DftiSetValue( b1, DFTI_INPUT_STRIDES, strides_out );
            status = DftiSetValue( b1, DFTI_OUTPUT_STRIDES, strides_in );

            status = DftiSetValue( f1, DFTI_NUMBER_OF_TRANSFORMS, rsize[2] );
            status = DftiSetValue( b1, DFTI_NUMBER_OF_TRANSFORMS, rsize[2] );

            status = DftiSetValue( f1, DFTI_INPUT_DISTANCE, 1 );
            status = DftiSetValue( b1, DFTI_INPUT_DISTANCE, 1 );

            status = DftiSetValue( f1, DFTI_OUTPUT_DISTANCE, 1 );
            status = DftiSetValue( b1, DFTI_OUTPUT_DISTANCE, 1 );

            status = DftiCommitDescriptor(f1);
            status = DftiCommitDescriptor(b1);
        }

        // In-place
        // Complex to complex along y direction
        // Repeated along x direction
        // Will need filter.z calls for each z
        {
            MKL_LONG strides[2]  = { 0, csize[2] };

            status = DftiCreateDescriptor( &t2, ZNN_DFTI_TYPE,
                                           DFTI_COMPLEX, 1, csize[1] );

            status = DftiSetValue( t2, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( t2, DFTI_PLACEMENT, DFTI_INPLACE );
            status = DftiSetValue( t2, DFTI_NUMBER_OF_TRANSFORMS, csize[0] );
            status = DftiSetValue( t2, DFTI_INPUT_STRIDES, strides );
            status = DftiSetValue( t2, DFTI_OUTPUT_STRIDES, strides );
            status = DftiSetValue( t2, DFTI_INPUT_DISTANCE, csize[2]*csize[1] );
            status = DftiSetValue( t2, DFTI_OUTPUT_DISTANCE, csize[2]*csize[1] );

            status = DftiCommitDescriptor(t2);
        }


        // In-place
        // Complex to complex along z direction
        // Repeated along x and y directions
        // Single call
        {
            MKL_LONG strides[2]  = { 0, 1 };

            status = DftiCreateDescriptor( &t3, ZNN_DFTI_TYPE,
                                           DFTI_COMPLEX, 1, csize[2] );

            status = DftiSetValue( t3, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( t3, DFTI_PLACEMENT, DFTI_INPLACE );
            status = DftiSetValue( t3, DFTI_NUMBER_OF_TRANSFORMS, csize[0]*csize[1] );
            status = DftiSetValue( t3, DFTI_INPUT_STRIDES, strides );
            status = DftiSetValue( t3, DFTI_OUTPUT_STRIDES, strides );
            status = DftiSetValue( t3, DFTI_INPUT_DISTANCE, csize[2] );
            status = DftiSetValue( t3, DFTI_OUTPUT_DISTANCE, csize[2] );

            status = DftiCommitDescriptor(t3);
        }
    }


    void forward( real* rp, void* cpv )
    {
        MKL_LONG status;

        real* cp = reinterpret_cast<real*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(real)*2);

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            status = DftiComputeForward(f1,
                                        rp + rsize[2] * i,
                                        cp + csize[2] * i * 2 );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            status = DftiComputeForward( t2, cp + i * 2, cp + i * 2 );
        }

        // In-place complex to complex along z-direction
        status = DftiComputeForward( t3, cp, cp );
    }

    void backward( void* cpv, real* rp )
    {
        MKL_LONG status;

        real* cp = reinterpret_cast<real*>(cpv);
        // In-place complex to complex along z-direction
        status = DftiComputeBackward( t3, cp, cp );

        // In-place complex to complex along y-direction
        // Care only about last rsize[2]
        long_t zOff = csize[2] - rsize[2];
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            status = DftiComputeBackward( t2,
                                          cp + (i + zOff)*2,
                                          cp + (i + zOff)*2 );
        }

        // Out-of-place complex to real along x-direction
        // Care only about last rsize[1] and rsize[2]
        long_t yOff = csize[1] - rsize[1];
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            status = DftiComputeBackward( b1,
                                          cp + (csize[2] * ( i + yOff ) + zOff) * 2,
                                          rp + rsize[2] * i );
        }
    }

    real get_scale() const
    {
        return scale;
    }

    size_t num_in_elements() const
    {
        return rsize[0] * rsize[1] * rsize[2];
    }

    size_t num_out_elements() const
    {
        return csize[0] * csize[1] * csize[2];
    }

    size_t in_memory() const
    {
        return num_in_elements() * sizeof(real);
    }

    size_t out_memory() const
    {
        return num_out_elements() * sizeof(real) * 2;
    }

    size_t memory() const
    {
        return in_memory() + out_memory();
    }


};

}} // namespace znn::fwd
