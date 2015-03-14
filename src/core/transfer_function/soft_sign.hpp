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

#ifndef ZNN_CORE_TRANSFER_FUNCTION_SOFT_SIGN_HPP_INCLUDED
#define ZNN_CORE_TRANSFER_FUNCTION_SOFT_SIGN_HPP_INCLUDED

#include <type_traits>
#include <cmath>

namespace zi {
namespace znn {
namespace functions {

struct soft_sign
{
public:
    double operator()(double x) const noexcept
    {
        return x / ( static_cast<double>(1) + std::abs(x) );
    }

    double grad(double f) const noexcept
    {
        return (static_cast<double>(1) - std::abs(f)) *
            (static_cast<double>(1) - std::abs(f));
    }
};

}}} // namespace zi::znn::functions


#endif // ZNN_CORE_TRANSFER_FUNCTION_SOFT_SIGN_HPP_INCLUDED
