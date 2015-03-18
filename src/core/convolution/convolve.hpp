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

#ifndef ZNN_CORE_CONVOLUTION_CONVOLVE_HPP_INCLUDED
#define ZNN_CORE_CONVOLUTION_CONVOLVE_HPP_INCLUDED

#include "../types.hpp"
#include "../utils.hpp"
#include "../volume_pool.hpp"
#include "convolve_constant.hpp"

namespace zi {
namespace znn {

template< typename T >
inline void convolve_add( const vol<T>& a, const vol<T>& b, vol<T>& r) noexcept
{
    if ( b.num_elements() == 1 )
    {
        convolve_constant_add(a,b.data()[0],r);
        return;
    }

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    ZI_ASSERT(r.shape()[0]==rx);
    ZI_ASSERT(r.shape()[1]==ry);
    ZI_ASSERT(r.shape()[2]==rz);

    for ( size_t x = 0; x < rx; ++x )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t z = 0; z < rz; ++z )
                for ( size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
                    for ( size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                        for ( size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                            r[x][y][z] += a[dx][dy][dz] * b[wx][wy][wz];
}

template< typename T >
inline void convolve( const vol<T>& a, const vol<T>& b, vol<T>& r) noexcept
{
    if ( b.num_elements() == 1 )
    {
        convolve_constant(a,b.data()[0],r);
    }

    fill(r,0);
    convolve_add(a,b,r);
}

template< typename T >
inline vol_p<T> convolve( const vol<T>& a, const vol<T>& b)
{
    vol_p<T> r = get_volume<T>(vec3i::one + size(a) - size(b));
    convolve(a,b,*r);
    return r;
}

template< typename T >
inline vol_p<T> convolve( const vol_p<T>& a, const vol_p<T>& b)
{
    return convolve(*a, *b);
}


template< typename T >
inline void
convolve_flipped_add( const vol<T>& a, const vol<T>& b, vol<T>& r) noexcept
{
    if ( size(a) == size(b) )
    {
        ZI_ASSERT(r.num_elements()==1);
        r.data()[0] += convolve_constant_flipped(a,b);
        return;
    }

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    ZI_ASSERT(r.shape()[0]==rx);
    ZI_ASSERT(r.shape()[1]==ry);
    ZI_ASSERT(r.shape()[2]==rz);

    for ( size_t x = 0; x < rx; ++x )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t z = 0; z < rz; ++z )
                for ( size_t dx = 0; dx < bx; ++dx )
                    for ( size_t dy = 0; dy < by; ++dy )
                        for ( size_t dz = 0; dz < bz; ++dz )
                            r[x][y][z] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
}



template< typename T >
inline void
convolve_flipped( const vol<T>& a, const vol<T>& b, vol<T>& r) noexcept
{
    if ( size(a) == size(b) )
    {
        ZI_ASSERT(r.num_elements()==1);
        r.data()[0] = convolve_constant_flipped(a,b);
        return;
    }

    fill(r,0);
    convolve_flipped_add(a,b,r);
}


template< typename T >
inline vol_p<T> convolve_flipped( const vol<T>& a, const vol<T>& b)
{
    vol_p<T> r = get_volume<T>(vec3i::one + size(a) - size(b));
    convolve_flipped(a,b,*r);
    return r;
}

template< typename T >
inline vol_p<T> convolve_flipped( const vol_p<T>& a, const vol_p<T>& b)
{
    return convolve_flipped(*a, *b);
}


template< typename T >
inline void
convolve_inverse_add( const vol<T>& a, const vol<T>& b, vol<T>& r) noexcept
{
    if ( size(b) == vec3i::one )
    {
        convolve_constant_inverse_add(a,b.data()[0],r);
        return;
    }

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    ZI_ASSERT(r.shape()[0]==ax + bx - 1);
    ZI_ASSERT(r.shape()[1]==ay + by - 1);
    ZI_ASSERT(r.shape()[2]==az + bz - 1);

    for ( size_t dx = 0; dx < bx; ++dx )
        for ( size_t dy = 0; dy < by; ++dy )
            for ( size_t dz = 0; dz < bz; ++dz )
            {
                size_t fx = bx - 1 - dx;
                size_t fy = by - 1 - dy;
                size_t fz = bz - 1 - dz;

                for ( size_t x = 0; x < ax; ++x )
                    for ( size_t y = 0; y < ay; ++y )
                        for ( size_t z = 0; z < az; ++z )
                        {
                            r[x+fx][y+fy][z+fz] += a[x][y][z] * b[dx][dy][dz];
                        }
            }
}

template< typename T >
inline void
convolve_inverse( const vol<T>& a, const vol<T>& b, vol<T>& r) noexcept
{
    if ( size(b) == vec3i::one )
    {
        convolve_constant_inverse(a,b.data()[0],r);
        return;
    }

    fill(r,0);
    convolve_inverse_add(a,b,r);
}


template< typename T >
inline vol_p<T> convolve_inverse( const vol<T>& a, const vol<T>& b)
{
    vol_p<T> r = get_volume<T>(size(a) + size(b) - vec3i::one);
    convolve_inverse(a,b,*r);
    return r;
}

template< typename T >
inline vol_p<T> convolve_inverse( const vol_p<T>& a, const vol_p<T>& b)
{
    return convolve_inverse(*a, *b);
}


}} // namespace zi::znn

#endif // ZNN_CORE_CONVOLUTION_CONVOLVE_HPP_INCLUDED
