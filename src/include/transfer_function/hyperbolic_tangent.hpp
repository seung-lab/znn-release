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

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>

namespace znn { namespace v4 { namespace functions {

struct hyperbolic_tangent
{
private:
    real a_ = 1;
    real b_ = 1;
    real b_over_a;

public:
    hyperbolic_tangent( real a = 1, real b = 1 )
        : a_(a), b_(b), b_over_a(b_/a_)
    {}

    real operator()(real x) const noexcept
    {
        return a_ * std::tanh( b_ * x );
    }

    real grad(real f) const noexcept
    {
        return b_over_a * ( a_ - f ) * ( a_ + f );
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "tanh").
            push("function_args", std::to_string(a_) + "," + std::to_string(b_));
        return ret;
    }

};

}}} // namespace znn::v4functions
