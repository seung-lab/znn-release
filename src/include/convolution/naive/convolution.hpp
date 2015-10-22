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

#include "../../types.hpp"
#include "../../cube/cube.hpp"

#include "../detail/constant.hpp"
#include "detail/convolve.hpp"
#include "detail/sparse.hpp"

namespace znn { namespace v4 {

template< typename T >
inline cube_p<T> convolve_sparse( cube<T> const & a,
                                  cube<T> const & b,
                                  vec3i   const & s )
{
    if ( b.num_elements() == 1 )
    {
        auto r = get_cube<T>(size(a));
        detail::convolve_constant(a, b.data()[0], *r);
        return r;
    }
    else if ( s == vec3i::one )
    {
        auto r = get_cube<T>(vec3i::one + size(a) - size(b));
        detail::pure_convolve(a, b, *r);
        return r;
    }
    else
    {
        auto r = get_cube<T>(size(a) - (size(b) - vec3i::one)*s);
        detail::convolve_sparse(a,b,s,*r);
        return r;
    }
}

template< typename T >
inline void convolve_sparse_add( cube<T> const & a,
                                 cube<T> const & b,
                                 vec3i   const & s,
                                 cube<T> & r ) noexcept
{
    if ( b.num_elements() == 1 )
    {
        detail::convolve_constant_add(a, b.data()[0], r);
    }
    else if ( s == vec3i::one )
    {
        detail::pure_convolve_add(a, b, r);
    }
    else
    {
        detail::convolve_sparse_add(a,b,s,r);
    }
}

template< typename T >
inline cube_p<T> convolve_sparse_flipped( cube<T> const & a,
                                          cube<T> const & b,
                                          vec3i   const & s )

{
    if ( size(a) == size(b) )
    {
        auto r = get_cube<T>(vec3i::one);
        r->data()[0] = detail::convolve_constant_flipped(a, b);
        return r;
    }
    else if ( s == vec3i::one )
    {
        auto r = get_cube<T>(vec3i::one + size(a) - size(b));
        detail::pure_convolve_flipped(a, b, *r);
        return r;
    }
    else
    {
        auto r = get_cube<T>( (size(a)-size(b)) / s + vec3i::one);
        detail::convolve_sparse_flipped(a,b,s,*r);
        return r;
    }
}

template< typename T >
inline cube_p<T> convolve_sparse_inverse( cube<T> const & a,
                                          cube<T> const & b,
                                          vec3i   const & s )
{
    if ( b.num_elements() == 1 )
    {
        auto r = get_cube<T>(size(a));
        detail::convolve_constant_inverse(a,b.data()[0],*r);
        return r;
    }
    else if ( s == vec3i::one )
    {
        auto r = get_cube<T>(size(a) + size(b) - vec3i::one);
        detail::pure_convolve_inverse(a,b,*r);
        return r;
    }
    else
    {
        auto r = get_cube<T>( size(a) + (size(b) - vec3i::one) * s );
        detail::convolve_sparse_inverse(a,b,s,*r);
        return r;
    }
}

template< typename T >
inline void convolve_sparse_inverse_add( cube<T> const & a,
                                         cube<T> const & b,
                                         vec3i   const & s,
                                         cube<T> & r ) noexcept
{
    if ( b.num_elements() == 1 )
    {
        detail::convolve_constant_add(a,b.data()[0],r);
    }
    else if ( s == vec3i::one )
    {
        detail::pure_convolve_inverse_add(a,b,r);
    }
    else
    {
        detail::convolve_sparse_inverse(a,b,s,r);
    }
}


}} // namespace znn::v4
