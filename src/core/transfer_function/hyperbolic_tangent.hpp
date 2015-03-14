//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_CORE_TRANSFER_FUNCTION_HYPERBOLIC_TANGENT_HPP_INCLUDED
#define ZNN_CORE_TRANSFER_FUNCTION_HYPERBOLIC_TANGENT_HPP_INCLUDED

#include <type_traits>
#include <cmath>

namespace zi {
namespace znn {
namespace functions {

struct hyperbolic_tangent
{
private:
    double a_ = 1;
    double b_ = 1;
    double b_over_a;

public:
    hyperbolic_tangent( double a = 1, double b = 1 )
        : a_(a), b_(b), b_over_a(b_/a_)
    {}

    double operator()(double x) const noexcept
    {
        return a_ * std::tanh( b_ * x );
    }

    double grad(double f) const noexcept
    {
        return b_over_a * ( a_ - f ) * ( a_ + f );
    }
};

}}} // namespace zi::znn::functions


#endif // ZNN_CORE_TRANSFER_FUNCTION_HYPERBOLIC_TANGENT_HPP_INCLUDED
