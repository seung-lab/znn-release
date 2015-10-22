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
inline void convolve_sparse( cube<T> const & a,
                             cube<T> const & b,
                             vec3i   const & s,
                             cube<T> const & r )
{
    ZI_ASSERT(size(r)==(size(a)-(size(b)-vec3i::one)*s));

    vec3i as = size(a);
    vec3i bs = size(b);
    vec3i rs = size(r);

    int a_strides[3]  = { s[2], as[2]*s[1], as[2]*as[1]*s[0] };
    int r_strides[3]  = { s[2], rs[2]*s[1], rs[2]*rs[1]*s[0] };

    // sparseness
    for (int xs=0; xs<s[0]; xs++)
        for (int ys=0; ys<s[1]; ys++)
            for (int zs=0; zs<s[2]; zs++)
            {
                vec3i in_size( (as[0]-1)/s[0] + (xs == 0 ? 1 : 0),
                               (as[1]-1)/s[1] + (ys == 0 ? 1 : 0),
                               (as[2]-1)/s[2] + (zs == 0 ? 1 : 0) );

                const T* in_ptr  = &(a[xs][ys][zs]);
                T* out_ptr = &(r[xs][ys][zs]);

#ifdef ZNN_USE_FLOATS
                int status = vslsConvExec( conv_plans.get(in_size, bs),
                                           in_ptr, a_strides,
                                           b.data(), NULL,
                                           out_ptr, r_strides);
#else
                int status = vsldConvExec( conv_plans.get(in_size, bs),
                                           in_ptr, a_strides,
                                           b.data(), NULL,
                                           out_ptr, r_strides);
#endif
            }
}


template< typename T >
inline void convolve_sparse_flipped( cube<T> const & a,
                                     cube<T> const & b,
                                     vec3i   const & s,
                                     cube<T> & r ) noexcept
{
    ZI_ASSERT(size(r)==((size(a)-size(b))/s+vec3i::one));

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rx = (ax - bx) / s[0] + 1;
    size_t ry = (ay - by) / s[1] + 1;
    size_t rz = (az - bz) / s[2] + 1;

    ZI_ASSERT(r.shape()[0]==rx);
    ZI_ASSERT(r.shape()[1]==ry);
    ZI_ASSERT(r.shape()[2]==rz);

    for ( size_t qx = 0, x = 0; qx < rx; ++qx, x += s[0] )
        for ( size_t qy = 0, y = 0; qy < ry; ++qy, y += s[1] )
            for ( size_t qz = 0, z = 0; qz < rz; ++qz, z += s[2] )
                for ( size_t dx = 0; dx < bx; ++dx )
                    for ( size_t dy = 0; dy < by; ++dy )
                        for ( size_t dz = 0; dz < bz; ++dz )
                            r[qx][qy][qz] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
}


template< typename T >
inline void convolve_sparse_inverse( cube<T> const & a,
                                     cube<T> const & b,
                                     vec3i   const & s,
                                     cube<T> const & r )
{
    ZI_ASSERT(size(r)==(size(a)+(size(b)-vec3i::one)*s));

    auto tmp = get_copy(b);
    flip(*tmp);

    vec3i as = size(a);
    vec3i bs = size(b);
    vec3i rs = size(r);

    int a_strides[3]  = { s[2], as[2]*s[1], as[2]*as[1]*s[0] };
    int r_strides[3]  = { s[2], rs[2]*s[1], rs[2]*rs[1]*s[0] };

    for (int xs=0; xs<s[0]; xs++)
        for (int ys=0; ys<s[1]; ys++)
            for (int zs=0; zs<s[2]; zs++)
            {
                vec3i in_size( (as[0]-1)/s[0] + (xs == 0),
                               (as[1]-1)/s[1] + (ys == 0),
                               (as[2]-1)/s[2] + (zs == 0) );

                const T* in_ptr  = &(a[xs][ys][zs]);
                T* out_ptr = &(r[xs][ys][zs]);

#ifdef ZNN_USE_FLOATS
                int status = vslsConvExec( conv_plans.get_inv(in_size, bs),
                                           in_ptr, a_strides,
                                           b.data(), NULL,
                                           out_ptr, r_strides );
#else
                int status = vsldConvExec( conv_plans.get_inv(in_size, bs),
                                           in_ptr, a_strides,
                                           b.data(), NULL,
                                           out_ptr, r_strides );
#endif

            }
}


}}} // namespace znn::v4::detail
