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

#include <memory>

#include "../../types.hpp"

namespace znn { namespace v4 {

template <typename T>
using cube = boost::multi_array<T, 3>;

template <typename T>
using qube = boost::multi_array<T, 4>;

template<typename T>
std::shared_ptr<cube<T>> get_cube(const vec3i& s)
{
    return std::shared_ptr<cube<T>>
        (new cube<T>(extents[s[0]][s[1]][s[2]]));
}

template<typename T>
std::shared_ptr<qube<T>> get_qube(const vec4i& s)
{
    return std::shared_ptr<qube<T>>
        (new qube<T>(extents[s[0]][s[1]][s[2]][s[3]]));
}

}} // namespace znn::v4
