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

#include "../types.hpp"
#include "convolve_mkl.hpp"

namespace znn { namespace v4 {


template< typename T >
inline void convolve_sparse_add( cube<T> const   & a,
                                 cube<T> const   & b,
                                 vec3i const     & s,
                                 cube<T>         & r ) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_add(a,b,r);
        return;
    }

    vec3i as = size(a);
    vec3i bs = size(b);
    vec3i rs = as - (bs - vec3i::one) * s;

    ZI_ASSERT(r.shape()[0]==rs[0]);
    ZI_ASSERT(r.shape()[1]==rs[1]);
    ZI_ASSERT(r.shape()[2]==rs[2]);

    vec3i tas = (as - vec3i::one) / s + vec3i::one;
    vec3i trs = tas + vec3i::one - bs;

    auto tap = get_cube<T>(tas);
    auto trp = get_cube<T>(trs);

    cube<T>& ta = *tap;
    cube<T>& tr = *trp;

    auto plan = conv_plans.get(tas, bs);

    // sparseness
    for (int xs=0; xs<s[0]; xs++)
        for (int ys=0; ys<s[1]; ys++)
            for (int zs=0; zs<s[2]; zs++)
            {
                // prepare input
                for (std::size_t x=xs, xt=0; x<as[0]; x+=s[0], ++xt)
                    for (std::size_t y=ys, yt=0; y<as[1]; y+=s[1], ++yt)
                        for(std::size_t z=zs, zt=0; z<as[2]; z+=s[2], ++zt)
                            ta[xt][yt][zt] = a[x][y][z];

#ifdef ZNN_USE_FLOATS
                int status = vslsConvExec( plan, ta.data(), NULL,
                                           b.data(), NULL, tr.data(), NULL);

#else
                int status = vsldConvExec( plan, ta.data(), NULL,
                                           b.data(), NULL, tr.data(), NULL);
#endif

                //std::cout << tr << std::endl;

                // combine subconvolution results
                for (std::size_t x=xs, wx=0; x<rs[0]; x+=s[0], ++wx)
                    for (std::size_t y=ys, wy=0; y<rs[1]; y+=s[1], ++wy )
                        for (std::size_t z=zs, wz=0; z<rs[2]; z+=s[2], ++wz)
                            r[x][y][z] += tr[wx][wy][wz];

            }

}


template< typename T >
inline cube_p<T> convolve_sparse( cube<T> const & a,
                                  cube<T> const & b,
                                  vec3i const & s )
{
    if ( s == vec3i::one )
    {
        return convolve(a,b);
    }

    cube_p<T> r = get_cube<T>(size(a) - (size(b) - vec3i::one) * s);

    fill(*r,0);
    convolve_sparse_add(a,b,s,*r);
    return r;
}

template< typename T >
inline cube_p<T> convolve_sparse( ccube_p<T> const & a,
                                  ccube_p<T> const & b,
                                  vec3i const & s )
{
    return convolve_sparse(*a,*b,s);
}


template< typename T >
inline void convolve_sparse_flipped_add( cube<T> const & a,
                                         cube<T> const & b,
                                         vec3i const & s,
                                         cube<T> & r ) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_flipped_add(a,b,r);
        return;
    }

    vec3i as = size(a);
    vec3i bs = size(b);
    vec3i rs = (as - bs) / s + vec3i::one;


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
inline void convolve_sparse_flipped( cube<T> const & a,
                                     cube<T> const & b,
                                     vec3i const & s,
                                     cube<T> & r ) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_flipped(a,b,r);
        return;
    }

    fill(r,0);
    convolve_sparse_flipped_add(a,b,s,r);
}


template< typename T >
inline cube_p<T> convolve_sparse_flipped( cube<T> const & a,
                                          cube<T> const & b,
                                          vec3i const & s )
{
    if ( s == vec3i::one )
    {
        return convolve_flipped(a,b);
    }

    ZI_ASSERT(((size(a)-size(b))/s)*s==(size(a)-size(b)));
    cube_p<T> r = get_cube<T>( (size(a)-size(b)) / s + vec3i::one);

    fill(*r,0);
    convolve_sparse_flipped_add(a,b,s,*r);
    return r;
}


template< typename T >
inline cube_p<T> convolve_sparse_inverse( cube<T> const & a,
                                          cube<T> const & borig,
                                          vec3i const & s )
{
    if ( s == vec3i::one )
    {
        return convolve_inverse(a,borig);
    }

    auto bp = get_copy(borig);
    cube<T>& b = *bp;
    flip(b);

    vec3i as = size(a);
    vec3i bs = size(b);
    vec3i rs = size(a) + (size(b) - vec3i::one) * s;

    cube_p<T> r = get_cube<T>(rs);

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
                T* out_ptr = &((*r)[xs][ys][zs]);

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
    return r;
}


template< typename T >
inline cube_p<T> convolve_sparse_inverse( ccube_p<T> const & a,
                                          ccube_p<T> const & b,
                                          vec3i const & s )
{
    return convolve_sparse_inverse(*a,*b,s);
}



template< typename T >
inline void convolve_sparse_inverse_add( cube<T> const & a,
                                         cube<T> const & b,
                                         vec3i const & s,
                                         cube<T> & r ) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_inverse_add(a,b,r);
        return;
    }

    auto radd = convolve_sparse_inverse(a,b,s);
    r += *radd;

}

}} // namespace znn::v4
