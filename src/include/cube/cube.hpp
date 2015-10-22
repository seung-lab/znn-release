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

#if   defined( DUMMY_CUBE )
#  include "detail/dummy_cube.hpp"
#elif defined( ZNN_MALLOC_CUBE )
#  include "detail/malloc_cube.hpp"
#elif defined( ZNN_CUBE_POOL_LOCKFREE )
#  include "detail/lockfree_cube_pool.hpp"
#elif defined( ZNN_CUBE_POOL)
#  include "detail/cube_pool.hpp"
#else
#  include "detail/dummy_cube.hpp"
#endif

namespace znn { namespace v4 {

#if 1

template <typename T> using ccube = cube<T>;
template <typename T> using cqube = qube<T>;

#else

template <typename T> using ccube = const cube<T>;
template <typename T> using cqube = const qube<T>;

#endif

template <typename T> using cube_p  = std::shared_ptr<cube<T>>;
template <typename T> using ccube_p = std::shared_ptr<ccube<T>>;

template <typename T> using qube_p  = std::shared_ptr<qube<T>>;
template <typename T> using cqube_p = std::shared_ptr<cqube<T>>;


template <typename T>
inline vec3i size( cube<T> const & a )
{
    return vec3i(a.shape()[0],a.shape()[1],a.shape()[2]);
};

template <typename T>
inline vec4i size( qube<T> const & a )
{
    return vec4i(a.shape()[0],a.shape()[1],a.shape()[2],a.shape()[3]);
};

template <typename T>
inline cube_p<T> get_copy( cube<T> const & c )
{
    auto r = get_cube<T>(size(c));
    *r = c;
    return r;
}

template<typename T>
inline cube_p<T> get_cube(long_t x, long_t y, long_t z)
{
    return get_cube<T>(vec3i(x,y,z));
}


}} // namespace znn::v4
