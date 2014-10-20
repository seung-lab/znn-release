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

#ifndef ZNN_COST_FN_HPP_INCLUDED
#define ZNN_COST_FN_HPP_INCLUDED

#include "../core/types.hpp"
#include "../core/volume_utils.hpp"

namespace zi {
namespace znn {

class cost_fn
{
public:
    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> outputs,
                                              std::list<double3d_ptr> labels,
                                              std::list<bool3d_ptr>   masks ) = 0;

    virtual double compute_cost( std::list<double3d_ptr> outputs,
                                 std::list<double3d_ptr> labels,
                                 std::list<bool3d_ptr>   masks ) = 0;

    virtual double compute_cls_error( std::list<double3d_ptr> outputs,
                                      std::list<double3d_ptr> labels,
                                      std::list<bool3d_ptr>   masks,
                                      double threshold = 0.5 )
    {
        double ret = static_cast<double>(0);
        std::list<double3d_ptr>::iterator lit = labels.begin();
        std::list<bool3d_ptr>::iterator mit = masks.begin();
        FOR_EACH( it, outputs )
        {
            double3d_ptr err = 
                volume_utils::classification_error(*it, *lit++, threshold);
            volume_utils::elementwise_masking(err, *mit++);
            ret += volume_utils::sum_all(err);
        }
        return ret;
    }

    virtual double get_output_number( bool3d_ptr mask )
    {
        return volume_utils::nnz(mask);
    }

    virtual double get_output_number( std::list<bool3d_ptr> masks )
    {
        return volume_utils::nnz(masks);
    }

    virtual void print_cost( double cost ) = 0;

    virtual void print_cls_error( double cls_err )
    {
        std::cout << "CLSE: " << cls_err;
    }

}; // abstract class cost_fn

typedef boost::shared_ptr<cost_fn> cost_fn_ptr;

}} // namespace zi::znn

#endif // ZNN_COST_FN_HPP_INCLUDED