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

#include <zi/utility/singleton.hpp>

#include "../types.hpp"
#include "../cube/cube.hpp"
#include "../cube/cube_operators.hpp"

namespace znn { namespace v4 {

class filter
{
protected:
    cube_p<real>   W_;
    cube_p<real>   mom_volume_;

    // weight update stuff
    real        eta_          = 0.1 ;
    real        momentum_     = 0.0 ;
    real        weight_decay_ = 0.0 ;

    // for shared filter
    std::mutex  mutex_;

public:
    typedef std::map<std::string, std::vector<std::shared_ptr<filter>>>
            pool_type;

    static pool_type &  shared_filters_pool;

protected:
    static real batch_size;

public:
    static void set_batch_size( real s )
    {
        ZI_ASSERT(s > 0);
        filter::batch_size = s;
    }

public:
    filter( vec3i const & sz, real eta, real mom = 0.0, real wd = 0.0 )
        : W_(get_cube<real>(sz))
        , mom_volume_(get_cube<real>(sz))
        , eta_(eta), momentum_(mom), weight_decay_(wd)
    {
	   fill(*mom_volume_,0);
    }

    real & eta()
    {
        return eta_;
    }

    cube<real >& W()
    {
        return *W_;
    }

    cube<real> & momentum_volume()
    {
        return *mom_volume_;
    }

    real & momentum()
    {
        return momentum_;
    }

    real & weight_decay()
    {
        return weight_decay_;
    }

public:
    void update( cube<real> const & dEdW ) noexcept
    {
        guard g(mutex_);

        real delta = -eta_/filter::batch_size;

        if ( momentum_ == 0 )
        {
            // W' = W - eta*dEdW/ps - eta*wd*W
            //    = W(1 - eta*wd)  - eta*dEdW/ps;

            if ( weight_decay_ != 0 )
            {
                *W_ *= static_cast<real>(1) - eta_ * weight_decay_;
            }

            mad_to( delta, dEdW, *W_ );
        }
        else
        {
            *mom_volume_ *= momentum_;
            mad_to( delta, dEdW, *mom_volume_ );

            if ( weight_decay_ != 0 )
            {
                mad_to( -eta_ * weight_decay_, *W_, *mom_volume_ );
            }

            *W_ += *mom_volume_;
        }
    }

}; // class filter

filter::pool_type& filter::shared_filters_pool =
        zi::singleton<filter::pool_type>::instance();

real filter::batch_size = 1;

}} // namespace znn::v4
