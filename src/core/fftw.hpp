//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
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

#ifndef ZNN_FFTW_HPP_INCLUDED
#define ZNN_FFTW_HPP_INCLUDED

#include "types.hpp"
#include "volume_pool.hpp"
#include "fftw_plans.hpp"
#include "utils.hpp"
#include "measure.hpp"

#include <zi/utility/assert.hpp>
#include <zi/concurrency.hpp>
#include <zi/time.hpp>

namespace zi {
namespace znn {

class zfft
{
public:
    static complex3d_ptr forward( const boost::shared_ptr<double3d>& in,
                                  vec3i pad = vec3i(0,0,0), bool measure = false )
    {
        PROFILE_FUNCTION();
        double3d_ptr  fft_in  ;
        complex3d_ptr fft_out ;
        fftw_plan     fft_plan;

        if ( pad[0] )
        {
            fft_in  = volume_pool.get_double3d(pad);
            fft_out = volume_pool.get_complex3d(pad);

            {
                zi::class_mutex<zfft>::guard g;

                fft_plan = fftw_plan_dft_r2c_3d
                    ( pad[0], pad[1], pad[2],
                      reinterpret_cast<double*>(fft_in->data()),
                      reinterpret_cast<fftw_complex*>(fft_out->data()),
                      measure ? FFTW_MEASURE : FFTW_ESTIMATE );
            }

            volume_utils::zero_out(fft_in);
            (*fft_in)[boost::indices[range(0,pad[0])][range(0,pad[1])][range(0,pad[2])]]
                = (*in);
        }
        else
        {
            fft_in  = volume_pool.get_double3d(in);
            fft_out = volume_pool.get_complex3d(in);

            {
                zi::class_mutex<zfft>::guard g;

                fft_plan = fftw_plan_dft_r2c_3d
                    ( fft_in->shape()[0], fft_in->shape()[1], fft_in->shape()[2],
                      reinterpret_cast<double*>(fft_in->data()),
                      reinterpret_cast<fftw_complex*>(fft_out->data()),
                      measure ? FFTW_MEASURE : FFTW_ESTIMATE );
            }

            (*fft_in) = (*in);
        }

        fftw_execute(fft_plan);
        fftw_destroy_plan(fft_plan);

        return fft_out;
    }

}; // zfft


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
        // if ( total_ % 10000 == 0 )
        // {
        //     std::cout << "Total spent on ffts: " << total_time_ << std::endl;
        // }
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

        zi::wall_timer wt;
        fftw_execute_dft_r2c(plan,
                             reinterpret_cast<double*>(in->data()),
                             reinterpret_cast<fftw_complex*>(out->data()));
        fftw_stats.add(wt.elapsed<double>());
    }

    static void backward( const boost::shared_ptr<complex3d>& in,
                          const boost::shared_ptr<double3d>& out )
    {
        PROFILE_FUNCTION();
        ZI_ASSERT(in->shape()[0]==out->shape()[0]);
        ZI_ASSERT(in->shape()[1]==out->shape()[1]);
        ZI_ASSERT((out->shape()[2]/2+1)==in->shape()[2]);

        fftw_plan plan = fftw_plans.get_backward(
            vec3i(out->shape()[0],out->shape()[1],out->shape()[2]));

        zi::wall_timer wt;
        fftw_execute_dft_c2r(plan,
                             reinterpret_cast<fftw_complex*>(in->data()),
                             reinterpret_cast<double*>(out->data()));
        fftw_stats.add(wt.elapsed<double>());
    }

    static complex3d_ptr forward( const boost::shared_ptr<double3d>& in )
    {
        PROFILE_FUNCTION();
        complex3d_ptr ret = volume_pool.get_complex3d(in);
        fftw::forward( in, ret );
        return ret;
    }

    static double3d_ptr backward( const complex3d_ptr& in,
                                  const vec3i& s )
    {
        PROFILE_FUNCTION();
        double3d_ptr ret = volume_pool.get_double3d(s);
        fftw::backward( in, ret );
        return ret;
    }

    static complex3d_ptr forward_pad( const double3d_ptr& in,
                                      const vec3i& pad )
    {
        PROFILE_FUNCTION();
        double3d_ptr pin = volume_pool.get_double3d(pad);
        volume_utils::zero_pad(pin, in);
        return fftw::forward(pin);
    }

}; // class fftw

class fft_transform
{
private:
    const size_t x, y, z;
    const boost::shared_ptr<double3d>  inb ;
    const boost::shared_ptr<complex3d> outb;
    fftw_plan forward_plan;
    fftw_plan backward_plan;

public:
    static std::size_t r2c_size( std::size_t z )
    {
        return (z/2)+1;
    }

public:
    fft_transform( const boost::shared_ptr<double3d>& in,
                   const boost::shared_ptr<complex3d>& out,
                   bool measure = true )
        : x(in->shape()[0])
        , y(in->shape()[1])
        , z(in->shape()[2])
    {
        zi::class_mutex<fft_transform>::guard g;
        forward_plan = fftw_plan_dft_r2c_3d
            ( x,y,z,
              reinterpret_cast<double*>(in->data()),
              reinterpret_cast<fftw_complex*>(out->data()),
              measure ? FFTW_MEASURE : FFTW_ESTIMATE );

        backward_plan = fftw_plan_dft_c2r_3d
            ( x,y,z,
              reinterpret_cast<fftw_complex*>(out->data()),
              reinterpret_cast<double*>(in->data()),
              measure ? FFTW_MEASURE : FFTW_ESTIMATE );

        ZI_ASSERT(in->shape()[0]==out->shape()[0]);
        ZI_ASSERT(in->shape()[1]==out->shape()[1]);
        ZI_ASSERT((in->shape()[2]/2+1)==out->shape()[2]);
    }

    ~fft_transform()
    {
        zi::class_mutex<fft_transform>::guard g;
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);
    }

    void forward()
    {
        fftw_execute(forward_plan);
    }

    void backward()
    {
        fftw_execute(backward_plan);
    }
};


}} // namespace zi::znn

#endif // ZNN_FFTW_HPP_INCLUDED