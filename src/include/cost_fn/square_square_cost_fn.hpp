//
// Copyright (C) 2014  Kisuk Lee <kisuklee@mit.edu>
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

#ifndef ZNN_SQUARE_SQUARE_COST_FN_HPP_INCLUDED
#define ZNN_SQUARE_SQUARE_COST_FN_HPP_INCLUDED

#include "cost_fn.hpp"
#include "../core/volume_pool.hpp"

namespace zi {
namespace znn {

class square_square_cost_fn: virtual public cost_fn
{
private:
    double  margin_;

public:
    virtual double3d_ptr gradient( double3d_ptr output,
                                   double3d_ptr label,
                                   bool3d_ptr   mask )
    {
        double3d_ptr ret = volume_pool.get_double3d(output);   

        double one  = static_cast<double>(1);
        double zero = static_cast<double>(0);

        std::size_t n = output->num_elements();
        for ( std::size_t i = 0; i < n; ++i )
        {
            if ( label->data()[i] == one )
            {
                ret->data()[i] = static_cast<double>(-2) *
                    std::max(one - margin_ - output->data()[i], zero);
            }
            else
            {
                ret->data()[i] = static_cast<double>(2) *
                    std::max(output->data()[i] - margin_, zero);
            }
        }
        
        volume_utils::elementwise_masking(ret, mask);
        return ret;
    }

    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> outputs,
                                              std::list<double3d_ptr> labels,
                                              std::list<bool3d_ptr>   masks )
    {
        std::list<double3d_ptr> ret;
        std::list<double3d_ptr>::iterator lit = labels.begin();
        std::list<bool3d_ptr>::iterator mit = masks.begin();
        FOR_EACH( it, outputs )
        {
            ret.push_back(gradient(*it, *lit++, *mit++));
        }
        return ret;
    }

    virtual double compute_cost( double3d_ptr output,
                                 double3d_ptr label,
                                 bool3d_ptr   mask )
    {
        // compute and return error
        double3d_ptr err = volume_pool.get_double3d(output);

        double one  = static_cast<double>(1);
        double zero = static_cast<double>(0);

        std::size_t n = output->num_elements();
        for ( std::size_t i = 0; i < n; ++i )
        {
            if ( label->data()[i] == one )
            {
                err->data()[i] = 
                    std::max(one - margin_ - output->data()[i], zero);
            }
            else
            {
                err->data()[i] = 
                    std::max(output->data()[i] - margin_, zero);
            }
        }               
        
        volume_utils::elementwise_masking(err, mask);
        return volume_utils::square_sum(err);
    }

    virtual double compute_cost( std::list<double3d_ptr> outputs, 
                                 std::list<double3d_ptr> labels,
                                 std::list<bool3d_ptr>   masks )
    {
        double ret = static_cast<double>(0);
        std::list<double3d_ptr>::iterator lit = labels.begin();
        std::list<bool3d_ptr>::iterator mit = masks.begin();
        FOR_EACH( it, outputs )
        {
            ret += compute_cost(*it, *lit++, *mit++);
        }
        return ret;
    }

    virtual void print_cost( double cost )
    {
        std::cout << "SQSQ (m=" << margin_ << "): " << cost;
    }


public:
    square_square_cost_fn( double margin = 0.1 )
        : margin_(margin)
    {}


}; // class square_square_cost_fn

}} // namespace zi::znn

#endif // ZNN_SQUARE_SQUARE_COST_FN_HPP_INCLUDED
