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

#include <string>
#include <type_traits>

namespace znn { namespace v4 { namespace functions {

struct linear
{
private:
    real a_ = 1;
    real b_ = 0;

public:
    linear( real a = 1, real b = 0 )
        : a_(a), b_(b)
    {}

    real operator()(real x) const noexcept
    {
        return a_ * x + b_;
    }

    real grad(real) const noexcept
    {
        return a_;
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "linear").
            push("function_args", std::to_string(a_) + "," + std::to_string(b_));
        return ret;
    }

};

}}} // namespace znn::v4::functions
