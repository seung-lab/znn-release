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

#ifndef ZNN_CROSS_ENTROPY_COST_FN_HPP_INCLUDED
#define ZNN_CROSS_ENTROPY_COST_FN_HPP_INCLUDED

#include "cost_fn.hpp"
#include "../core/volume_pool.hpp"

#include <iomanip>

namespace zi {
namespace znn {

class cross_entropy_cost_fn: virtual public cost_fn
{
public:
    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> probs,
                                              std::list<double3d_ptr> labels,
                                              std::list<bool3d_ptr>   masks )
    {
        std::list<double3d_ptr> ret;
        std::list<double3d_ptr>::iterator lit = labels.begin();
        std::list<bool3d_ptr>::iterator   mit = masks.begin();
        FOR_EACH( it, probs )
        {
            double3d_ptr grad = volume_pool.get_double3d(*it);
            volume_utils::sub_from_mul(grad,*it,*lit++,1);
            volume_utils::elementwise_masking(grad,*mit++);
            ret.push_back(grad);
        }
        return ret;
    }

    virtual double compute_cost( std::list<double3d_ptr> probs,
                                 std::list<double3d_ptr> labels,
                                 std::list<bool3d_ptr>   masks )
    {
        // each mask in the masks list is assumed to be the same
        FOR_EACH( it, masks )
        {
            STRONG_ASSERT(*(masks.front()) == **it);
        }

        double3d_ptr err = volume_utils::cross_entropy(probs,labels);
        volume_utils::elementwise_masking(err,masks.front());
        return volume_utils::sum_all(err);
    }

    virtual double compute_cls_error( std::list<double3d_ptr> probs,
                                      std::list<double3d_ptr> labels,
                                      std::list<bool3d_ptr>   masks,
                                      double thresh = 0.5 )
    {
        // [11/13/2013 kisuklee]
        // Should modify to handle multi-class classification error
        double3d_ptr err = volume_utils::classification_error(probs.back(),labels.back(),thresh);
        volume_utils::elementwise_masking(err,masks.back());
        return volume_utils::sum_all(err);
    }

    virtual double get_output_number( std::list<bool3d_ptr> masks )
    {
        // each mask in the masks list is assumed to be the same
        FOR_EACH( it, masks )
        {
            STRONG_ASSERT(*(masks.front()) == **it);
        }

        return cost_fn::get_output_number(masks.back());
    }

    virtual void print_cost( double cost )
    {
        std::cout << "logprob: " << cost;
    }

}; // class cross_entropy_cost_fn

}} // namespace zi::znn

#endif // ZNN_CROSS_ENTROPY_COST_FN_HPP_INCLUDED
