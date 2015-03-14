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

#ifndef ZNN_CORE_TRANSFER_FUNCTION_RECTIFY_LINEAR_HPP_INCLUDED
#define ZNN_CORE_TRANSFER_FUNCTION_RECTIFY_LINEAR_HPP_INCLUDED

#include <type_traits>
#include <algorithm>

namespace zi {
namespace znn {
namespace functions {

struct rectify_linear
{
public:
    double operator()(double x) const noexcept
    {
        return std::max( static_cast<double>(0), x );
    }

    double grad(double f) const noexcept
    {
        return ( f > 0 ) ? 1 : 0;
    }
};

}}} // namespace zi::znn::functions


#endif // ZNN_CORE_TRANSFER_FUNCTION_RECTIFY_LINEAR_HPP_INCLUDED
