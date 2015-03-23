//
// Copyright (C) 2014-present  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_CORE_FFT_FFTW_PLANS_HPP_INCLUDED
#define ZNN_CORE_FFT_FFTW_PLANS_HPP_INCLUDED

#include "../types.hpp"
#include "../volume_pool.hpp"

#include <zi/concurrency.hpp>
#include <zi/utility/assert.hpp>
#include <zi/utility/for_each.hpp>
#include <zi/utility/singleton.hpp>
#include <zi/time/time.hpp>

#include <map>
#include <iostream>

#define ZNN_FFTW_PLANNING_MODE FFTW_PATIENT

namespace zi {
namespace znn {

class fftw_plans_impl
{
private:
    zi::mutex                  m_         ;
    std::map<vec3i, fftw_plan> fwd_       ;
    std::map<vec3i, fftw_plan> bwd_       ;
    double                     time_      ;

public:
    ~fftw_plans_impl()
    {
        FOR_EACH( it, fwd_ )
        {
            fftw_destroy_plan(it->second);
        }

        FOR_EACH( it, bwd_ )
        {
            fftw_destroy_plan(it->second);
        }
    }

    fftw_plans_impl()
        : m_()
        , fwd_()
        , bwd_()
        , time_(0)
    { }

    fftw_plan get_forward( const vec3i& s )
    {
        zi::mutex::guard g(m_);
        if ( fwd_.count(s) )
        {
            return fwd_[s];
        }

        zi::wall_timer wt;
        wt.reset();

        double3d_ptr   in  = volume_pool.get_double3d(s);
        complex3d_ptr  out = volume_pool.get_complex3d(s);

        fftw_plan ret = fftw_plan_dft_r2c_3d
            ( s[0], s[1], s[2],
              reinterpret_cast<double*>(in->data()),
              reinterpret_cast<fftw_complex*>(out->data()),
              ZNN_FFTW_PLANNING_MODE );

        fwd_[s] = ret;
        time_ += wt.elapsed<double>();

        std::cout << "Total time spent creating fftw plans: "
                  << time_ << std::endl;

        return ret;
    }

    fftw_plan get_backward( const vec3i& s )
    {
        zi::mutex::guard g(m_);
        if ( bwd_.count(s) )
        {
            return bwd_[s];
        }

        zi::wall_timer wt;
        wt.reset();

        complex3d_ptr  in  = volume_pool.get_complex3d(s);
        double3d_ptr   out = volume_pool.get_double3d(s);

        fftw_plan ret = fftw_plan_dft_c2r_3d
            ( s[0], s[1], s[2],
              reinterpret_cast<fftw_complex*>(in->data()),
              reinterpret_cast<double*>(out->data()),
              ZNN_FFTW_PLANNING_MODE );

        bwd_[s] = ret;
        time_ += wt.elapsed<double>();

        std::cout << "Total time spent creating fftw plans: "
                  << time_ << std::endl;

        return ret;
    }

}; // class fftw_plans_impl

namespace {
fftw_plans_impl& fftw_plans =
    zi::singleton<fftw_plans_impl>::instance();
} // anonymous namespace


}} // namespace zi::znn

#endif // ZNN_CORE_FFT_FFTW_PLANS_HPP_INCLUDED
