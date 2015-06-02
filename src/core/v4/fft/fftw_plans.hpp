#pragma once

#include "../types.hpp"
#include "../cube/cube.hpp"

#include <zi/utility/singleton.hpp>
#include <zi/time/time.hpp>

#include <map>
#include <iostream>
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
typedef fftwf_plan    fft_plan   ;
typedef fftwf_complex fft_complex;

#else

#define FFT_DESTROY_PLAN fftw_destroy_plan
#define FFT_CLEANUP      fftw_cleanup
#define FFT_PLAN_C2R     fftw_plan_dft_c2r_3d
#define FFT_PLAN_R2C     fftw_plan_dft_r2c_3d
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


class fft_plans_impl
{
private:
    std::mutex                m_          ;
    std::map<vec3i, fft_plan> fwd_        ;
    std::map<vec3i, fft_plan> bwd_        ;
    real                      time_       ;

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
