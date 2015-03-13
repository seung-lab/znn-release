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

#ifndef ZNN_CORE_CONVOLUTION_CONVOLVE_SPARSE_HPP_INCLUDED
#define ZNN_CORE_CONVOLUTION_CONVOLVE_SPARSE_HPP_INCLUDED

#include "../types.hpp"
#include "../utils.hpp"
#include "../volume_pool.hpp"
#include "convolve.hpp"

namespace zi {
namespace znn {

template< typename T >
inline void convolve_sparse_add( const vol<T>& a, const vol<T>& b,
                                 const vec3i& s, vol<T>& r) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_add(a,b,r);
        return;
    }

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax - rbx + 1;
    size_t ry = ay - rby + 1;
    size_t rz = az - rbz + 1;

    ZI_ASSERT(r.shape()[0]==rx);
    ZI_ASSERT(r.shape()[1]==ry);
    ZI_ASSERT(r.shape()[2]==rz);

    for ( size_t x = 0; x < rx; ++x )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t z = 0; z < rz; ++z )
                for ( size_t dx = x, wx = bx-1; dx < rbx + x; dx += s[0], --wx )
                    for ( size_t dy = y, wy = by-1; dy < rby + y; dy += s[1], --wy )
                        for ( size_t dz = z, wz = bz-1; dz < rbz + z; dz += s[2], --wz )
                            r[x][y][z] += a[dx][dy][dz] * b[wx][wy][wz];

}

template< typename T >
inline void convolve_sparse( const vol<T>& a, const vol<T>& b,
                             const vec3i& s, vol<T>& r) noexcept
{
    if ( s == vec3i::one )
    {
        convolve(a,b,r);
        return;
    }

    fill(r,0);
    convolve_sparse_add(a,b,s,r);
}


template< typename T >
inline vol_p<T>
convolve_sparse( const vol<T>& a, const vol<T>& b, const vec3i& s)
{
    if ( s == vec3i::one )
    {
        return convolve(a,b);
    }

    vol_p<T> r =
        get_volume<T>(size(a) - (size(b) - vec3i::one) * s);

    fill(*r,0);
    convolve_sparse_add(a,b,s,*r);
    return r;
}

template< typename T >
inline vol_p<T>
convolve_sparse( const vol_p<T>& a, const vol_p<T>& b, const vec3i& s)
{
    return convolve_sparse(*a,*b,s);
}

template< typename T >
inline void convolve_sparse_flipped_add( const vol<T>& a, const vol<T>& b,
                                         const vec3i& s, vol<T>& r) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_flipped_add(a,b,r);
        return;
    }

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
inline void convolve_sparse_flipped( const vol<T>& a, const vol<T>& b,
                                     const vec3i& s, vol<T>& r) noexcept
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
inline vol_p<T> convolve_sparse_flipped( const vol<T>& a, const vol<T>& b,
                                         const vec3i& s)
{
    if ( s == vec3i::one )
    {
        return convolve_flipped(a,b);
    }

    vol_p<T> r = get_volume<T>( (size(a)-size(b)) / s + vec3i::one);

    fill(*r,0);
    convolve_sparse_flipped_add(a,b,s,*r);
    return r;
}


template< typename T >
inline vol_p<T> convolve_sparse_flipped( const vol_p<T>& a, const vol_p<T>& b,
                                         const vec3i& s)
{
    return convolve_sparse_flipped(*a,*b,s);
}



template< typename T >
inline void convolve_sparse_inverse_add( const vol<T>& a, const vol<T>& b,
                                         const vec3i& s, vol<T>& r) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_inverse_add(a,b,r);
        return;
    }

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax + rbx - 1;
    size_t ry = ay + rby - 1;
    size_t rz = az + rbz - 1;

    ZI_ASSERT(r.shape()[0]==rx);
    ZI_ASSERT(r.shape()[1]==ry);
    ZI_ASSERT(r.shape()[2]==rz);

    for ( size_t wx = 0; wx < bx; ++wx )
        for ( size_t wy = 0; wy < by; ++wy )
            for ( size_t wz = 0; wz < bz; ++wz )
            {
                size_t fx = bx - 1 - wx;
                size_t fy = by - 1 - wy;
                size_t fz = bz - 1 - wz;

                size_t ox = fx * s[0];
                size_t oy = fy * s[1];
                size_t oz = fz * s[2];

                for ( size_t x = 0; x < ax; ++x )
                    for ( size_t y = 0; y < ay; ++y )
                        for ( size_t z = 0; z < az; ++z )
                            r[x+ox][y+oy][z+oz] += a[x][y][z] * b[wx][wy][wz];
            }
}


template< typename T >
inline void convolve_sparse_inverse( const vol<T>& a, const vol<T>& b,
                                     const vec3i& s, vol<T>& r) noexcept
{
    if ( s == vec3i::one )
    {
        convolve_inverse(a,b,r);
        return;
    }

    fill(r,0);
    convolve_sparse_inverse_add(a,b,s,r);
}


template< typename T >
inline vol_p<T> convolve_sparse_inverse( const vol<T>& a, const vol<T>& b,
                                         const vec3i& s)
{
    if ( s == vec3i::one )
    {
        return convolve_inverse(a,b);
    }

    vol_p<T> r = get_volume<T>( size(a) + (size(b) - vec3i::one) * s );
    fill(*r,0);
    convolve_sparse_inverse_add(a,b,s,*r);
    return r;
}


template< typename T >
inline vol_p<T> convolve_sparse_inverse( const vol_p<T>& a, const vol_p<T>& b,
                                         const vec3i& s)
{
    return convolve_sparse_inverse(*a,*b,s);
}

}} // namespace zi::znn

#endif // ZNN_CORE_CONVOLUTION_CONVOLVE_SPARSE_HPP_INCLUDED
