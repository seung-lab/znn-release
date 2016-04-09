//
// Copyright (C)      2016  Kisuk Lee           <kisuklee@mit.edu>
//
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

#include "types.hpp"

#include <algorithm>

namespace znn { namespace v4 {

inline vec3i minimum( vec3i const & a, vec3i const & b )
{
    vec3i r;

    r[0] = std::min(a[0],b[0]);
    r[1] = std::min(a[1],b[1]);
    r[2] = std::min(a[2],b[2]);

    return r;
}

inline vec3i maximum( vec3i const & a, vec3i const & b )
{
    vec3i r;

    r[0] = std::max(a[0],b[0]);
    r[1] = std::max(a[1],b[1]);
    r[2] = std::max(a[2],b[2]);

    return r;
}

}} // namespace znn::v4