#pragma once

#include "fftw_plans.hpp"

#include <zi/time.hpp>

namespace znn { namespace v4 {

class fftw_stats_impl
{
private:
    dboule              total_time_;
    std::size_t         total_     ;
    mutable std::mutex  m_         ;

public:
    fftw_stats_impl()
        : total_time_(0)
        , total_(0)
        , m_()
    { }

    dboule get_total_time() const
    {
        guard g(m_);
        return total_time_;
    }

    void reset_total_time()
    {
        guard g(m_);
        total_time_ = 0;
    }

    size_t get_total() const
    {
        guard g(m_);
        return total_;
    }

    void add(dboule time)
    {
        guard g(m_);
        ++total_;
        total_time_ += time;
    }
};

namespace {
fftw_stats_impl& fftw_stats = zi::singleton<fftw_stats_impl>::instance();
} // anonymous namespace

class fftw
{
public:
    class transformer
    {
    private:
        vec3i     sz           ;
        fftw_plan forward_plan ;
        fftw_plan backward_plan;

    public:
        transformer(const vec3i& s)
            : sz(s)
            , forward_plan(fftw_plans.get_forward(s))
            , backward_plan(fftw_plans.get_backward(s))
        {}

        void forward( cube<dboule>& in,
                      cube<complex>& out )
        {
            ZI_ASSERT(size(out)==fft_complex_size(in));
            ZI_ASSERT(size(in)==sz);

#           ifdef MEASURE_FFT_RUNTIME
            zi::wall_timer wt;
#           endif
            fftw_execute_dft_r2c(forward_plan,
                                 reinterpret_cast<dboule*>(in.data()),
                                 reinterpret_cast<fftw_complex*>(out.data()));
#           ifdef MEASURE_FFT_RUNTIME
            fftw_stats.add(wt.elapsed<dboule>());
#           endif
        }

        void backward( cube<complex>& in,
                       cube<dboule>& out )
        {
            ZI_ASSERT(size(in)==fft_complex_size(out));
            ZI_ASSERT(size(out)==sz);

#           ifdef MEASURE_FFT_RUNTIME
            zi::wall_timer wt;
#           endif
            fftw_execute_dft_c2r(backward_plan,
                                 reinterpret_cast<fftw_complex*>(in.data()),
                                 reinterpret_cast<dboule*>(out.data()));
#           ifdef MEASURE_FFT_RUNTIME
            fftw_stats.add(wt.elapsed<dboule>());
#           endif
        }

        cube_p<complex> forward( cube_p<dboule>&& in )
        {
            cube_p<complex> ret = get_cube<complex>(fft_complex_size(*in));
            forward( *in, *ret );
            return ret;
        }

        cube_p<complex> forward_pad( const ccube_p<dboule>& in )
        {
            cube_p<dboule> pin = pad_zeros(*in, sz);
            return forward(std::move(pin));
        }

        cube_p<dboule> backward( cube_p<complex>&& in )
        {
            cube_p<dboule> ret = get_cube<dboule>(sz);
            backward( *in, *ret );
            return ret;
        }
    };


public:
    static void forward( cube<dboule>& in,
                         cube<complex>& out )
    {
        ZI_ASSERT(in.shape()[0]==out.shape()[0]);
        ZI_ASSERT(in.shape()[1]==out.shape()[1]);
        ZI_ASSERT((in.shape()[2]/2+1)==out.shape()[2]);

        fftw_plan plan = fftw_plans.get_forward(
            vec3i(in.shape()[0],in.shape()[1],in.shape()[2]));

#       ifdef MEASURE_FFT_RUNTIME
        zi::wall_timer wt;
#       endif
        fftw_execute_dft_r2c(plan,
                             reinterpret_cast<dboule*>(in.data()),
                             reinterpret_cast<fftw_complex*>(out.data()));
#       ifdef MEASURE_FFT_RUNTIME
        fftw_stats.add(wt.elapsed<dboule>());
#       endif
    }

    static void backward( cube<complex>& in,
                          cube<dboule>& out )
    {
        ZI_ASSERT(in.shape()[0]==out.shape()[0]);
        ZI_ASSERT(in.shape()[1]==out.shape()[1]);
        ZI_ASSERT((out.shape()[2]/2+1)==in.shape()[2]);

        fftw_plan plan = fftw_plans.get_backward(
            vec3i(out.shape()[0],out.shape()[1],out.shape()[2]));

#       ifdef MEASURE_FFT_RUNTIME
        zi::wall_timer wt;
#       endif
        fftw_execute_dft_c2r(plan,
                             reinterpret_cast<fftw_complex*>(in.data()),
                             reinterpret_cast<dboule*>(out.data()));
#       ifdef MEASURE_FFT_RUNTIME
        fftw_stats.add(wt.elapsed<dboule>());
#       endif
    }

    static cube_p<complex> forward( cube_p<dboule>&& in )
    {
        cube_p<complex> ret = get_cube<complex>(fft_complex_size(*in));
        fftw::forward( *in, *ret );
        return ret;
    }

    static cube_p<dboule> backward( cube_p<complex>&& in, const vec3i& s )
    {
        cube_p<dboule> ret = get_cube<dboule>(s);
        fftw::backward( *in, *ret );
        return ret;
    }

    static cube_p<complex> forward_pad( const ccube_p<dboule>& in,
                                        const vec3i& pad )
    {
        cube_p<dboule> pin = pad_zeros(*in, pad);
        return fftw::forward(std::move(pin));
    }

}; // class fftw

}} // namespace znn::v4
