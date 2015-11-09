//
// Copyright (C) 2015  Kisuk Lee <kisuklee@mit.edu>
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

#ifndef ZNN_MALIS_COST_FN_HPP_INCLUDED
#define ZNN_MALIS_COST_FN_HPP_INCLUDED

#include "cost_fn.hpp"
#include "../core/volume_pool.hpp"
#include "malis.hpp"

namespace zi {
namespace znn {

class malis_cost_fn: virtual public cost_fn
{
private:
    // margin
    double                  m_;

    // to avoid duplicate call of malis()
    std::list<malis_metric> unique_queue_;


public:
    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> outputs,
                                              std::list<double3d_ptr> labels,
                                              std::list<bool3d_ptr>   masks )
    {
        // compute malis (gradient,loss) pair
        malis_pair mp = malis(labels, outputs, masks, m_);
        
        // store loss value for later use
        unique_queue_.clear();
        unique_queue_.push_back(mp.second);

        // return gradient
        return mp.first;
    }

    virtual double compute_cost( std::list<double3d_ptr> outputs, 
                                 std::list<double3d_ptr> labels,
                                 std::list<bool3d_ptr>   masks )
    {
        if ( unique_queue_.empty() )
        {
            malis_pair mp = malis(labels, outputs, masks, m_);
            unique_queue_.push_back(mp.second);
        }

        malis_metric metric = unique_queue_.back();
        unique_queue_.clear();

        double n = get_output_number(masks);
        return n*metric.loss;
    }

    virtual void print_cost( double cost )
    {
        std::cout << "malis loss (m=" << m_ << "): " << cost;
    }


public:
    malis_cost_fn(double m = 0)
        : m_(m)
        , unique_queue_()
    {}

}; // class malis_cost_fn

}} // namespace zi::znn

#endif // ZNN_MALIS_COST_FN_HPP_INCLUDED
