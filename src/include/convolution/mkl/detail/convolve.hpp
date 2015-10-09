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

#include "plans.hpp"
#include "../../../cube/cube_operators.hpp"

namespace znn { namespace v4 { namespace detail {

template< typename T >
inline void convolve( cube<T> const & a,
                      cube<T> const & b,
                      cube<T> & r) noexcept
{
    ZI_ASSERT(size(r)==(vec3i::one+size(a)-size(b)));

#ifdef ZNN_USE_FLOATS
    int status = vslsConvExec(conv_plans.get(size(a),size(b)),
                              a.data(), NULL,
                              b.data(), NULL,
                              r.data(), NULL);
#else
    int status = vsldConvExec(conv_plans.get(size(a),size(b)),
                              a.data(), NULL,
                              b.data(), NULL,
                              r.data(), NULL);
#endif
}

template< typename T >
inline void convolve_flipped( cube<T> const & a,
                              cube<T> const & b,
                              cube<T> const & r ) noexcept
{
    ZI_ASSERT(size(r)==(vec3i::one+size(a)-size(b)));

    auto tmp = get_copy(a);
    flip(*tmp);
    return convolve(*tmp,b,r);
}

template< typename T >
inline void convolve_inverse( cube<T> const & a,
                              cube<T> const & b,
                              cube<T> const & r) noexcept
{
    ZI_ASSERT(size(r)==(size(a)+size(b)-vec3i::one));

    auto tmp = get_copy(b);
    flip(*tmp);

#ifdef ZNN_USE_FLOATS
    int status = vslsConvExec(conv_plans.get_inv(size(a),size(b)),
                              a.data(), NULL,
                              tmp->data(), NULL,
                              r.data(), NULL);
#else
    int status = vsldConvExec(conv_plans.get_inv(size(a),size(b)),
                              a.data(), NULL,
                              tmp->data(), NULL,
                              r.data(), NULL);
}

}}} // namespace znn::v4::detail
