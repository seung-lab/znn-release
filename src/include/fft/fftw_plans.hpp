//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#ifdef ZNN_USE_MKL_FFT
#  include <fftw/fftw3.h>
#else
#  include <fftw3.h>
#endif

#include "../types.hpp"
#include "../cube/cube.hpp"

#include <zi/utility/singleton.hpp>
#include <zi/time/time.hpp>

#include <map>
#include <iostream>
#include <unordered_map>
#include <type_traits>
#include <mutex>

#ifndef ZNN_FFTW_PLANNING_MODE
#  define ZNN_FFTW_PLANNING_MODE (FFTW_ESTIMATE)
#endif

namespace znn { namespace v4 {

#ifdef ZNN_USE_FLOATS

#define FFT_DESTROY_PLAN fftwf_destroy_plan
#define FFT_CLEANUP      fftwf_cleanup
#define FFT_PLAN_C2R     fftwf_plan_dft_c2r_3d
#define FFT_PLAN_R2C     fftwf_plan_dft_r2c_3d

#define FFT_PLAN_MANY_DFT fftwf_plan_many_dft
#define FFT_PLAN_MANY_R2C fftwf_plan_many_dft_r2c
#define FFT_PLAN_MANY_C2R fftwf_plan_many_dft_c2r

#define FFT_EXECUTE_DFT_R2C fftwf_execute_dft_r2c
#define FFT_EXECUTE_DFT_C2R fftwf_execute_dft_c2r
#define FFT_EXECUTE_DFT     fftwf_execute_dft

typedef fftwf_plan    fft_plan   ;
typedef fftwf_complex fft_complex;

#else

#define FFT_DESTROY_PLAN fftw_destroy_plan
#define FFT_CLEANUP      fftw_cleanup
#define FFT_PLAN_C2R     fftw_plan_dft_c2r_3d
#define FFT_PLAN_R2C     fftw_plan_dft_r2c_3d

#define FFT_PLAN_MANY_DFT fftw_plan_many_dft
#define FFT_PLAN_MANY_R2C fftw_plan_many_dft_r2c
#define FFT_PLAN_MANY_C2R fftw_plan_many_dft_c2r

#define FFT_EXECUTE_DFT_R2C fftw_execute_dft_r2c
#define FFT_EXECUTE_DFT_C2R fftw_execute_dft_c2r
#define FFT_EXECUTE_DFT     fftw_execute_dft

typedef fftw_plan    fft_plan   ;
typedef fftw_complex fft_complex;

#endif

inline vec3i fft_complex_size(const vec3i& s)
{
    auto r = s;
    r[2] /= 2;
    r[2] += 1;
    return r;
}

template< typename T >
inline vec3i fft_complex_size(const cube<T>& c)
{
    return fft_complex_size(size(c));
}

class fft_image_plan
{
private:
    vec3i rsize, csize;

    fft_plan f1, f2, f3;
    fft_plan b1, b2, b3;

public:
    ~fft_image_plan()
    {
        FFT_DESTROY_PLAN(f1);
        FFT_DESTROY_PLAN(f2);
        FFT_DESTROY_PLAN(f3);
        FFT_DESTROY_PLAN(b1);
        FFT_DESTROY_PLAN(b2);
        FFT_DESTROY_PLAN(b3);
    }

    fft_image_plan( vec3i const & _rsize, vec3i const & _size )
        : rsize(_rsize)
    {

        ZI_ASSERT(_size[0]==_rsize[0]);

        csize = _size;
        csize[0] /= 2; csize[0] += 1;

        auto rv = get_cube<real>(rsize);
        auto cv = get_cube<complex>(fft_complex_size(csize));

        real*        rp = reinterpret_cast<real*>(rv->data());
        fft_complex* cp = reinterpret_cast<fft_complex*>(cv->data());

        // Out-of-place
        // Real to complex / complex to real along x direction
        // Repeated along z direction
        // Will need input.y calls for each y
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
        // Will need output.z calls for each z
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
    }


    cube_p<complex> forward( cube<real> & im )
    {
        auto ret = get_cube<complex>(csize);

        // TODO: dometimes this is not needed
        fill(*ret,0);

        real*        rp = reinterpret_cast<real*>(im.data());
        fft_complex* cp = reinterpret_cast<fft_complex*>(ret->data());


        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            FFT_EXECUTE_DFT_R2C( f1,
                                 rp + rsize[2] * i,
                                 cp + csize[2] * i );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            FFT_EXECUTE_DFT( f2, cp + i, cp + i );

        }

        // In-place complex to complex along x-direction
        FFT_EXECUTE_DFT( f3, cp, cp );

        return ret;
    }

    cube_p<real> backward( cube<complex> & im )
    {
        auto ret = get_cube<real>(rsize);

        real*        rp = reinterpret_cast<real*>(ret->data());
        fft_complex* cp = reinterpret_cast<fft_complex*>(im.data());

        // In-place complex to complex along x-direction
        FFT_EXECUTE_DFT( b3, cp, cp );

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            FFT_EXECUTE_DFT( b2, cp + i, cp + i );

        }

        // Out-of-place complex to real along x-direction
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            FFT_EXECUTE_DFT_C2R( b1,
                                 cp + csize[2] * i,
                                 rp + rsize[2] * i );
        }

        return ret;
    }

};


class fft_filter_plan
{
private:
    vec3i filter, sparse, size, rsize, csize;

    fft_plan f1, f2, f3;
    fft_plan b1, b2, b3;

public:
    ~fft_filter_plan()
    {
        FFT_DESTROY_PLAN(f1);
        FFT_DESTROY_PLAN(f2);
        FFT_DESTROY_PLAN(f3);
        FFT_DESTROY_PLAN(b1);
        FFT_DESTROY_PLAN(b2);
        FFT_DESTROY_PLAN(b3);
    }

    fft_filter_plan( vec3i const & _filter,
                     vec3i const & _sparse,
                     vec3i const & _size )
        : filter(_filter), sparse(_sparse), size(_size)
    {
        rsize = filter;
        rsize[0] = size[0];

        csize = size;
        csize[0] /= 2; csize[0] += 1;

        auto rv = get_cube<real>(rsize);
        auto cv = get_cube<complex>(fft_complex_size(csize));

        real*        rp = reinterpret_cast<real*>(rv->data());
        fft_complex* cp = reinterpret_cast<fft_complex*>(cv->data());

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
            int odist   = static_cast<int>(sparse[2]);

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
    }


    cube_p<complex> forward( cube<real> & im )
    {
        auto ret = get_cube<complex>(csize);
        auto rim = sparse_explode_x_slow( im, sparse[0], rsize[0] );

        fill(*ret,0);

        real*        rp = reinterpret_cast<real*>(rim->data());
        fft_complex* cp = reinterpret_cast<fft_complex*>(ret->data());


        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            FFT_EXECUTE_DFT_R2C( f1,
                                 rp + rsize[2] * i,
                                 cp + csize[2] * i * sparse[1] );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            FFT_EXECUTE_DFT( f2,
                             cp + i * sparse[2],
                             cp + i * sparse[2] );

        }

        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( f3, cp, cp );

        return ret;
    }

    cube_p<real> backward( cube<complex> & im )
    {
        auto ret = get_cube<real>(rsize);

        real*        rp = reinterpret_cast<real*>(ret->data());
        fft_complex* cp = reinterpret_cast<fft_complex*>(im.data());

        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( b3, cp, cp );

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            FFT_EXECUTE_DFT( b2,
                             cp + i * sparse[2],
                             cp + i * sparse[2] );

        }

        // Out-of-place complex to real along x-direction
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            FFT_EXECUTE_DFT_C2R( b1,
                                 cp + csize[2] * i * sparse[1],
                                 rp + rsize[2] * i );
        }

        ret = sparse_implode_x_slow( *ret, sparse[0], filter[0] );
        return ret;
    }

};

class fft_plans_impl
{
private:
    std::mutex                                           m_          ;
    std::unordered_map<vec3i, fft_plan, vec_hash<vec3i>> fwd_        ;
    std::unordered_map<vec3i, fft_plan, vec_hash<vec3i>> bwd_        ;
    real                                                 time_       ;



    static_assert(std::is_pointer<fft_plan>::value,
                  "fftw_plan must be a pointer");

public:
    ~fft_plans_impl()
    {
        for ( auto& p: fwd_ ) FFT_DESTROY_PLAN(p.second);
        for ( auto& p: bwd_ ) FFT_DESTROY_PLAN(p.second);
        FFT_CLEANUP();
    }

    fft_plans_impl(): m_(), fwd_(), bwd_(), time_(0)
    {
    }

    fft_plan get_forward( const vec3i& s )
    {
        guard g(m_);

        fft_plan& ret = fwd_[s];

        if ( ret ) return ret;

        zi::wall_timer wt; wt.reset();

        auto in  = get_cube<real>(s);
        auto out = get_cube<complex>(fft_complex_size(s));

        ret = FFT_PLAN_R2C
            ( s[0], s[1], s[2],
              reinterpret_cast<real*>(in->data()),
              reinterpret_cast<fft_complex*>(out->data()),
              ZNN_FFTW_PLANNING_MODE );

        time_ += wt.elapsed<real>();

        return ret;
    }

    fft_plan get_backward( const vec3i& s )
    {
        guard g(m_);

        fft_plan& ret = bwd_[s];

        if ( ret ) return ret;

        zi::wall_timer wt; wt.reset();

        auto in  = get_cube<complex>(fft_complex_size(s));
        auto out = get_cube<real>(s);

        ret = FFT_PLAN_C2R
            ( s[0], s[1], s[2],
              reinterpret_cast<fft_complex*>(in->data()),
              reinterpret_cast<real*>(out->data()),
              ZNN_FFTW_PLANNING_MODE );

        time_ += wt.elapsed<real>();

//        std::cout << "Total time spent creating fftw plans: "
//                  << time_ << std::endl;

        return ret;
    }

}; // class fft_plans_impl

namespace {
fft_plans_impl& fft_plans =
    zi::singleton<fft_plans_impl>::instance();
} // anonymous namespace


}} // namespace znn::v4

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
