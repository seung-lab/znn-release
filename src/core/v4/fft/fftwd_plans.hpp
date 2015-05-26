#pragma once

#include "../types.hpp"
#include "../cube/cube.hpp"

#include <zi/utility/singleton.hpp>
#include <zi/time/time.hpp>

#include <map>
#include <iostream>
#include <type_traits>
#include <mutex>

#define ZNN_FFTW_PLANNING_MODE (FFTW_ESTIMATE)

namespace znn { namespace v4 {

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


class fftw_plans_impl
{
private:
    std::mutex                 m_          ;
    std::map<vec3i, fftw_plan> fwd_        ;
    std::map<vec3i, fftw_plan> bwd_        ;
    dboule                     time_       ;

    static_assert(std::is_pointer<fftw_plan>::value,
                  "fftw_plan must be a pointer");

public:
    ~fftw_plans_impl()
    {
        for ( auto& p: fwd_ ) fftw_destroy_plan(p.second);
        for ( auto& p: bwd_ ) fftw_destroy_plan(p.second);
        fftw_cleanup();
    }

    fftw_plans_impl(): m_(), fwd_(), bwd_(), time_(0)
    {
    }

    fftw_plan get_forward( const vec3i& s )
    {
        guard g(m_);

        fftw_plan& ret = fwd_[s];

        if ( ret ) return ret;

        zi::wall_timer wt; wt.reset();

        auto in  = get_cube<dboule>(s);
        auto out = get_cube<complex>(fft_complex_size(s));

        ret = fftw_plan_dft_r2c_3d
            ( s[0], s[1], s[2],
              reinterpret_cast<dboule*>(in->data()),
              reinterpret_cast<fftw_complex*>(out->data()),
              ZNN_FFTW_PLANNING_MODE );

        time_ += wt.elapsed<dboule>();

//        std::cout << "Total time spent creating fftw plans: "
//                  << time_ << std::endl;

        return ret;
    }

    fftw_plan get_backward( const vec3i& s )
    {
        guard g(m_);

        fftw_plan& ret = bwd_[s];

        if ( ret ) return ret;

        zi::wall_timer wt; wt.reset();

        auto in  = get_cube<complex>(fft_complex_size(s));
        auto out = get_cube<dboule>(s);

        ret = fftw_plan_dft_c2r_3d
            ( s[0], s[1], s[2],
              reinterpret_cast<fftw_complex*>(in->data()),
              reinterpret_cast<dboule*>(out->data()),
              ZNN_FFTW_PLANNING_MODE );

        time_ += wt.elapsed<dboule>();

//        std::cout << "Total time spent creating fftw plans: "
//                  << time_ << std::endl;

        return ret;
    }

}; // class fftw_plans_impl

namespace {
fftw_plans_impl& fftw_plans =
    zi::singleton<fftw_plans_impl>::instance();
} // anonymous namespace


}} // namespace znn::v4
