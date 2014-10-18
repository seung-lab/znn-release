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

#ifndef ZNN_RECTIFY_LINEAR_ERROR_FN_HPP_INCLUDED
#define ZNN_RECTIFY_LINEAR_ERROR_FN_HPP_INCLUDED

#include "error_fn.hpp"
#include "../core/volume_pool.hpp"

#include <algorithm>
#include <utility>

namespace zi {
namespace znn {

class rectify_linear_error_fn: virtual public error_fn
{
public:
    virtual double3d_ptr gradient(double3d_ptr dEdF, double3d_ptr F)
    {
        std::size_t n = F->num_elements();
        double3d_ptr r = volume_pool.get_double3d(F);
        for ( std::size_t i = 0; i < n; ++i )
        {
            r->data()[i] = dEdF->data()[i] *
                ((F->data()[i] > 0) ?
                 static_cast<double>(1) : static_cast<double>(0));
        }
        return r;
    }

    virtual void apply(double3d_ptr v)
    {
        std::size_t n = v->shape()[0]*v->shape()[1]*v->shape()[2];
        for ( std::size_t i = 0; i < n; ++i )
        {
            v->data()[i] = std::max(static_cast<double>(0), v->data()[i]);
        }
    }

    virtual void add_apply(double c, double3d_ptr v)
    {
        std::size_t n = v->shape()[0]*v->shape()[1]*v->shape()[2];
        for ( std::size_t i = 0; i < n; ++i )
        {
            v->data()[i] = std::max(static_cast<double>(0), c + v->data()[i]);
        }
    }

}; // class rectify_linear_error_fn

}} // namespace zi::znn

#endif // ZNN_RECTIFY_LINEAR_ERROR_FN_HPP_INCLUDED
