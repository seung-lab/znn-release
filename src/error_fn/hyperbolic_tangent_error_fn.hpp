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

#ifndef ZNN_HYPERBOLIC_TANGENT_ERROR_FN_HPP_INCLUDED
#define ZNN_HYPERBOLIC_TANGENT_ERROR_FN_HPP_INCLUDED

#include "error_fn.hpp"
#include "../core/volume_pool.hpp"

namespace zi {
namespace znn {

class hyperbolic_tangent_error_fn: virtual public error_fn
{
private:
    double  a_;  // a*tanh(b*x)
    double  b_;  // a*tanh(b*x)

public:
    virtual double3d_ptr gradient(double3d_ptr dEdF, double3d_ptr F)
    {
        std::size_t n = F->shape()[0]*F->shape()[1]*F->shape()[2];
        double3d_ptr r = volume_pool.get_double3d(F->shape()[0],
                                                  F->shape()[1],
                                                  F->shape()[2]);
        for ( std::size_t i = 0; i < n; ++i )
        {
            r->data()[i] = dEdF->data()[i] * (b_/a_) * 
                (a_ - F->data()[i])*(a_ + F->data()[i]);
        }
        return r;
    }

    virtual void apply(double3d_ptr v)
    {
        std::size_t n = v->shape()[0]*v->shape()[1]*v->shape()[2];
        for ( std::size_t i = 0; i < n; ++i )
        {            
            v->data()[i] = a_*std::tanh(b_*v->data()[i]);
        }
    }

    virtual void add_apply(double c, double3d_ptr v)
    {
        std::size_t n = v->shape()[0]*v->shape()[1]*v->shape()[2];
        for ( std::size_t i = 0; i < n; ++i )
        {
            v->data()[i] = a_*std::tanh(b_*(c + v->data()[i]));
        }
    }

public:
    // hyperbolic_tangent_error_fn(double a = 1.7159, double b = 0.6666)
    hyperbolic_tangent_error_fn(double a = static_cast<double>(1),
                                double b = static_cast<double>(1))
        : a_(a)
        , b_(b)
    {
        // std::cout << a << "*tanh(" << b << " * x)" << std::endl;
    }

}; // class hyperbolic_tangent_error_fn

}} // namespace zi::znn

#endif // ZNN_HYPERBOLIC_TANGENT_ERROR_FN_HPP_INCLUDED
