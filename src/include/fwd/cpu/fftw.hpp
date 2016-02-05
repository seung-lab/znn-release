#pragma once

#ifdef ZNN_USE_MKL_FFTW
#  include <fftw/fftw3.h>
#else
#  include <fftw3.h>
#endif

#include <map>
#include <iostream>
#include <unordered_map>
#include <type_traits>
#include <mutex>
#include <zi/utility/singleton.hpp>

#include "../assert.hpp"
#include "../types.hpp"

#ifndef ZNN_FFTW_PLANNING_MODE
#  define ZNN_FFTW_PLANNING_MODE (FFTW_ESTIMATE)
#endif

namespace znn { namespace fwd {

#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)

#  define FFT_DESTROY_PLAN fftwl_destroy_plan
#  define FFT_CLEANUP      fftwl_cleanup
#  define FFT_PLAN_C2R     fftwl_plan_dft_c2r_3d
#  define FFT_PLAN_R2C     fftwl_plan_dft_r2c_3d

#  define FFT_PLAN_MANY_DFT fftwl_plan_many_dft
#  define FFT_PLAN_MANY_R2C fftwl_plan_many_dft_r2c
#  define FFT_PLAN_MANY_C2R fftwl_plan_many_dft_c2r

#  define FFT_EXECUTE_DFT_R2C fftwl_execute_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftwl_execute_dft_c2r
#  define FFT_EXECUTE_DFT     fftwl_execute_dft

typedef fftwl_plan    fft_plan   ;
typedef fftwl_complex fft_complex;

#elif defined(ZNN_USE_DOUBLE_PRECISION)

#  define FFT_DESTROY_PLAN fftw_destroy_plan
#  define FFT_CLEANUP      fftw_cleanup
#  define FFT_PLAN_C2R     fftw_plan_dft_c2r_3d
#  define FFT_PLAN_R2C     fftw_plan_dft_r2c_3d

#  define FFT_PLAN_MANY_DFT fftw_plan_many_dft
#  define FFT_PLAN_MANY_R2C fftw_plan_many_dft_r2c
#  define FFT_PLAN_MANY_C2R fftw_plan_many_dft_c2r

#  define FFT_EXECUTE_DFT_R2C fftw_execute_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftw_execute_dft_c2r
#  define FFT_EXECUTE_DFT     fftw_execute_dft

typedef fftw_plan    fft_plan   ;
typedef fftw_complex fft_complex;

#else

#  define FFT_DESTROY_PLAN fftwf_destroy_plan
#  define FFT_CLEANUP      fftwf_cleanup
#  define FFT_PLAN_C2R     fftwf_plan_dft_c2r_3d
#  define FFT_PLAN_R2C     fftwf_plan_dft_r2c_3d

#  define FFT_PLAN_MANY_DFT fftwf_plan_many_dft
#  define FFT_PLAN_MANY_R2C fftwf_plan_many_dft_r2c
#  define FFT_PLAN_MANY_C2R fftwf_plan_many_dft_c2r

#  define FFT_EXECUTE_DFT_R2C fftwf_execute_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftwf_execute_dft_c2r
#  define FFT_EXECUTE_DFT     fftwf_execute_dft

typedef fftwf_plan    fft_plan   ;
typedef fftwf_complex fft_complex;

#endif

class fft_transformer
{
private:
    vec3i rsize, csize;

    fft_plan f1, f2, f3;
    fft_plan b1, b2, b3;

    real scale;

public:
    ~fft_transformer()
    {
        FFT_DESTROY_PLAN(f1);
        FFT_DESTROY_PLAN(f2);
        FFT_DESTROY_PLAN(f3);
        FFT_DESTROY_PLAN(b1);
        FFT_DESTROY_PLAN(b2);
        FFT_DESTROY_PLAN(b3);
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

        real*        rp = new real[rsize[0]*rsize[1]*rsize[2]];
        fft_complex* cp = new fft_complex[csize[0]*csize[1]*csize[2]];

        // Out-of-place
        // Real to complex / complex to real along x direction
        // Repeated along z direction
        // Will need filter.y calls for each y
        {
            int n[]     = { static_cast<int>(rsize[0]) };
            int howmany = static_cast<int>(rsize[2]);
            int istride = static_cast<int>(rsize[1] * rsize[2]);
            int idist   = static_cast<int>(1);
            int ostride = static_cast<int>(csize[1] * csize[2]);
            int odist   = static_cast<int>(1);

            f1 = FFT_PLAN_MANY_R2C( 1, n, howmany,
                                    rp, NULL, istride, idist,
                                    cp, NULL, ostride, odist,
                                    ZNN_FFTW_PLANNING_MODE );

            b1 = FFT_PLAN_MANY_C2R( 1, n, howmany,
                                    cp, NULL, ostride, odist,
                                    rp, NULL, istride, idist,
                                    ZNN_FFTW_PLANNING_MODE );

        }

        // In-place
        // Complex to complex along y direction
        // Repeated along x direction
        // Will need filter.z calls for each z
        {
            int n[]     = { static_cast<int>(csize[1]) };
            int howmany = static_cast<int>(csize[0]);
            int stride  = static_cast<int>(csize[2]);
            int dist    = static_cast<int>(csize[2]*csize[1]);

            f2 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                    cp, NULL, stride, dist,
                                    cp, NULL, stride, dist,
                                    FFTW_FORWARD, ZNN_FFTW_PLANNING_MODE );

            b2 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                    cp, NULL, stride, dist,
                                    cp, NULL, stride, dist,
                                    FFTW_BACKWARD, ZNN_FFTW_PLANNING_MODE );

        }


        // In-place
        // Complex to complex along z direction
        // Repeated along x and y directions
        // Single call
        {
            int n[]     = { static_cast<int>(csize[2]) };
            int howmany = static_cast<int>(csize[0]*csize[1]);
            int stride  = static_cast<int>(1);
            int dist    = static_cast<int>(csize[2]);

            f3 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                    cp, NULL, stride, dist,
                                    cp, NULL, stride, dist,
                                    FFTW_FORWARD, ZNN_FFTW_PLANNING_MODE );

            b3 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                    cp, NULL, stride, dist,
                                    cp, NULL, stride, dist,
                                    FFTW_BACKWARD, ZNN_FFTW_PLANNING_MODE );

        }


        delete [] rp;
        delete [] cp;
    }


    void forward( real* rp, void* cpv )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(fft_complex));

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            FFT_EXECUTE_DFT_R2C( f1, rp + rsize[2] * i, cp + csize[2] * i );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            FFT_EXECUTE_DFT( f2, cp + i, cp + i );
        }

        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( f3, cp, cp );
    }

    void backward( void* cpv, real* rp )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( b3, cp, cp );

        // In-place complex to complex along y-direction
        // Care only about last rsize[2]
        long_t zOff = csize[2] - rsize[2];
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            FFT_EXECUTE_DFT( b2, cp + i + zOff, cp + i + zOff );
        }

        // Out-of-place complex to real along x-direction
        // Care only about last rsize[1] and rsize[2]
        long_t yOff = csize[1] - rsize[1];
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            FFT_EXECUTE_DFT_C2R( b1,
                                 cp + csize[2] * ( i + yOff ) + zOff,
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
        return num_out_elements() * sizeof(fft_complex);
    }

    size_t memory() const
    {
        return in_memory() + out_memory();
    }


};

}} // namespace znn::fwd

#undef FFT_DESTROY_PLAN
#undef FFT_CLEANUP
#undef FFT_PLAN_R2C
#undef FFT_PLAN_C2R

#undef FFT_PLAN_MANY_DFT
#undef FFT_PLAN_MANY_R2C
#undef FFT_PLAN_MANY_C2R

#undef FFT_EXECUTE_DFT_R2C
#undef FFT_EXECUTE_DFT_C2R
#undef FFT_EXECUTE_DFT
