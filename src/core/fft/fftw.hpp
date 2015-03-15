//
// Copyright (C) 2014-present  Aleksandar Zlateski <zlateski@mit.edu>
//                             Kisuk Lee           <kisuklee@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_FFT_FFTW_HPP_INCLUDED
#define ZNN_CORE_FFT_FFTW_HPP_INCLUDED

#include "fftw_plans.hpp"
#include "../types.hpp"
#include "../volume_pool.hpp"
#include "../volume_utils.hpp"
#include "../utils.hpp"
#include "../measure.hpp"

#include <zi/utility/assert.hpp>
#include <zi/concurrency.hpp>
#include <zi/time.hpp>

namespace zi {
namespace znn {

class fftw_stats_impl
{
private:
    double         total_time_;
    std::size_t    total_     ;
    zi::mutex      m_         ;

public:
    fftw_stats_impl()
        : total_time_(0)
        , total_(0)
        , m_()
    { }

    // [kisuklee]
    std::size_t get_total_time() const
    {
        return total_time_;
    }

    // [kisuklee]
    void reset_total_time()
    {
        total_time_ = 0;
    }

    void add(double time)
    {
        zi::mutex::guard g(m_);
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
    static void forward( const boost::shared_ptr<double3d>& in,
                         const boost::shared_ptr<complex3d>& out )
    {
        PROFILE_FUNCTION();
        ZI_ASSERT(in->shape()[0]==out->shape()[0]);
        ZI_ASSERT(in->shape()[1]==out->shape()[1]);
        ZI_ASSERT((in->shape()[2]/2+1)==out->shape()[2]);

        fftw_plan plan = fftw_plans.get_forward(
            vec3i(in->shape()[0],in->shape()[1],in->shape()[2]));

#ifdef MEASURE_FFT_RUNTIME
        zi::wall_timer wt;
#endif
        fftw_execute_dft_r2c(plan,
                             reinterpret_cast<double*>(in->data()),
                             reinterpret_cast<fftw_complex*>(out->data()));
#ifdef MEASURE_FFT_RUNTIME
        fftw_stats.add(wt.elapsed<double>());
#endif
    }

    static void backward( const boost::shared_ptr<complex3d>& in,
                          const boost::shared_ptr<double3d>& out )
    {
        ZI_ASSERT(in->shape()[0]==out->shape()[0]);
        ZI_ASSERT(in->shape()[1]==out->shape()[1]);
        ZI_ASSERT((out->shape()[2]/2+1)==in->shape()[2]);

        fftw_plan plan = fftw_plans.get_backward(
            vec3i(out->shape()[0],out->shape()[1],out->shape()[2]));

#ifdef MEASURE_FFT_RUNTIME
        zi::wall_timer wt;
#endif
        fftw_execute_dft_c2r(plan,
                             reinterpret_cast<fftw_complex*>(in->data()),
                             reinterpret_cast<double*>(out->data()));
#ifdef MEASURE_FFT_RUNTIME
        fftw_stats.add(wt.elapsed<double>());
#endif
    }

    static complex3d_ptr forward( const boost::shared_ptr<double3d>& in )
    {
        complex3d_ptr ret = volume_pool.get_complex3d(in);
        fftw::forward( in, ret );
        return ret;
    }

    static double3d_ptr backward( const complex3d_ptr& in,
                                  const vec3i& s )
    {
        double3d_ptr ret = volume_pool.get_double3d(s);
        fftw::backward( in, ret );
        return ret;
    }

    static complex3d_ptr forward_pad( const double3d_ptr& in,
                                      const vec3i& pad )
    {
        double3d_ptr pin = volume_pool.get_double3d(pad);
        volume_utils::zero_pad(pin, in);
        return fftw::forward(pin);
    }

}; // class fftw


}} // namespace zi::znn

#endif // ZNN_CORE_FFT_FFTW_HPP_INCLUDED
