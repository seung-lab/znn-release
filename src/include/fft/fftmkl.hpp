#pragma once

#include "fftmkl_plans.hpp"

#include <zi/time.hpp>

namespace znn { namespace v4 {

class fft_stats_impl
{
private:
    real              total_time_;
    std::size_t         total_     ;
    mutable std::mutex  m_         ;

public:
    fft_stats_impl()
        : total_time_(0)
        , total_(0)
        , m_()
    { }

    real get_total_time() const
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

    void add(real time)
    {
        guard g(m_);
        ++total_;
        total_time_ += time;
    }
};

namespace {
fft_stats_impl& fft_stats = zi::singleton<fft_stats_impl>::instance();
} // anonymous namespace

class fftw
{
public:
    class transformer
    {
    private:
        vec3i    sz           ;
        vec3i    actual_sz    ;
        fft_plan forward_plan ;
        fft_plan backward_plan;

        int32_t get_optimal(int32_t s) const
        {
            //return s;
            // SLOW so what, SUE ME!
            if ( s < 10 ) return s;
            if ( s > 384 ) return s;

            std::vector<int32_t> mins({
                    10,15,16,20,25,32,36,44,48,64,72,78,84,88,90,
                        100,112,128,140,150,160,176,200,220,224,
                        256,260,300,320,384});

            return *std::lower_bound(mins.begin(), mins.end(),s);
        }

    public:
        transformer(const vec3i& s)
            : sz(s)
            , actual_sz(s)
        {
            if ( s[1] == s[2] )
            {

            }

            forward_plan  = fft_plans.get_forward(actual_sz);
            backward_plan = fft_plans.get_backward(actual_sz);
        }

        vec3i const & size() const
        {
            return sz;
        }

        vec3i const & actual_size() const
        {
            return actual_sz;
        }

        void forward( cube<real>& in,
                      cube<complex>& out )
        {
            ZI_ASSERT(size(out)==fft_complex_size(in));
            ZI_ASSERT(size(in)==actual_sz);

            MKL_LONG status;

#           ifdef MEASURE_FFT_RUNTIME
            zi::wall_timer wt;
#           endif
            status = DftiComputeForward(*forward_plan,
                                        reinterpret_cast<real*>(in.data()),
                                        reinterpret_cast<real*>(out.data()));
#           ifdef MEASURE_FFT_RUNTIME
            fft_stats.add(wt.elapsed<real>());
#           endif
        }

        void backward( cube<complex>& in,
                       cube<real>& out )
        {
            ZI_ASSERT(size(in)==fft_complex_size(out));
            ZI_ASSERT(size(out)==actual_sz);

            MKL_LONG status;

#           ifdef MEASURE_FFT_RUNTIME
            zi::wall_timer wt;
#           endif
            status = DftiComputeBackward(*backward_plan,
                                         reinterpret_cast<real*>(in.data()),
                                         reinterpret_cast<real*>(out.data()));
#           ifdef MEASURE_FFT_RUNTIME
            fft_stats.add(wt.elapsed<real>());
#           endif
        }

        cube_p<complex> forward( cube_p<real>&& in )
        {
            cube_p<complex> ret = get_cube<complex>(fft_complex_size(*in));
            forward( *in, *ret );
            return ret;
        }

        cube_p<complex> forward_pad( const ccube_p<real>& in )
        {
            cube_p<real> pin = pad_zeros(*in, actual_sz);
            return forward(std::move(pin));
        }

        cube_p<real> backward( cube_p<complex>&& in )
        {
            cube_p<real> ret = get_cube<real>(actual_sz);
            backward( *in, *ret );
            return ret;
        }
    };


public:
    static void forward( cube<real>& in,
                         cube<complex>& out )
    {
        ZI_ASSERT(in.shape()[0]==out.shape()[0]);
        ZI_ASSERT(in.shape()[1]==out.shape()[1]);
        ZI_ASSERT((in.shape()[2]/2+1)==out.shape()[2]);

        fft_plan plan = fft_plans.get_forward(
            vec3i(in.shape()[0],in.shape()[1],in.shape()[2]));

        MKL_LONG status;

#       ifdef MEASURE_FFT_RUNTIME
        zi::wall_timer wt;
#       endif
        status = DftiComputeForward(*plan,
                                    reinterpret_cast<real*>(in.data()),
                                    reinterpret_cast<real*>(out.data()));
#       ifdef MEASURE_FFT_RUNTIME
        fft_stats.add(wt.elapsed<real>());
#       endif
    }

    static void backward( cube<complex>& in,
                          cube<real>& out )
    {
        ZI_ASSERT(in.shape()[0]==out.shape()[0]);
        ZI_ASSERT(in.shape()[1]==out.shape()[1]);
        ZI_ASSERT((out.shape()[2]/2+1)==in.shape()[2]);

        fft_plan plan = fft_plans.get_backward(
            vec3i(out.shape()[0],out.shape()[1],out.shape()[2]));

        MKL_LONG status;

#       ifdef MEASURE_FFT_RUNTIME
        zi::wall_timer wt;
#       endif
        status = DftiComputeBackward(*plan,
                                     reinterpret_cast<real*>(in.data()),
                                     reinterpret_cast<real*>(out.data()));
#       ifdef MEASURE_FFT_RUNTIME
        fft_stats.add(wt.elapsed<real>());
#       endif
    }

    static cube_p<complex> forward( cube_p<real>&& in )
    {
        cube_p<complex> ret = get_cube<complex>(fft_complex_size(*in));
        fftw::forward( *in, *ret );
        return ret;
    }

    static cube_p<real> backward( cube_p<complex>&& in, const vec3i& s )
    {
        cube_p<real> ret = get_cube<real>(s);
        fftw::backward( *in, *ret );
        return ret;
    }

    static cube_p<complex> forward_pad( const ccube_p<real>& in,
                                        const vec3i& pad )
    {
        cube_p<real> pin = pad_zeros(*in, pad);
        return fftw::forward(std::move(pin));
    }

}; // class fftw

}} // namespace znn::v4
