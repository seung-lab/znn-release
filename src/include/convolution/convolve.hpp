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

#ifdef ZNN_USE_MKL_DIRECT_CONV
#  include "convolve_mkl.hpp"
#else

#include "../types.hpp"
#include "convolve_constant.hpp"

namespace znn { namespace v4 {

// cube
template< typename T >
inline void convolve_add( cube<T> const & a,
                          cube<T> const & b,
                          cube<T> & r) noexcept
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

// qube
template< typename T >
inline void convolve_add( qube<T> const & a,
                          cube<T> const & b,
                          qube<T> & r) noexcept
{
    if ( b.num_elements() == 1 )
    {
        convolve_constant_add(a,b.data()[0],r);
        return;
    }

    size_t am = a.shape()[0];
    size_t ax = a.shape()[1];
    size_t ay = a.shape()[2];
    size_t az = a.shape()[3];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rm = am;
    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    ZI_ASSERT(r.shape()[0]==rm);
    ZI_ASSERT(r.shape()[1]==rx);
    ZI_ASSERT(r.shape()[2]==ry);
    ZI_ASSERT(r.shape()[3]==rz);

    for ( size_t m = 0; m < rm; ++m )
        for ( size_t x = 0; x < rx; ++x )
            for ( size_t y = 0; y < ry; ++y )
                for ( size_t z = 0; z < rz; ++z )
                    for ( size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
                        for ( size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                            for ( size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                                r[m][x][y][z] += a[m][dx][dy][dz] * b[wx][wy][wz];
}

// cube
template< typename T >
inline void convolve( cube<T> const & a,
                      cube<T> const & b,
                      cube<T> & r) noexcept
{
    if ( b.num_elements() == 1 )
    {
        convolve_constant(a,b.data()[0],r);
        return;
    }

    fill(r,0);
    convolve_add(a,b,r);
}

// qube
template< typename T >
inline void convolve( qube<T> const & a,
                      cube<T> const & b,
                      qube<T> & r) noexcept
{
    if ( b.num_elements() == 1 )
    {
        convolve_constant(a,b.data()[0],r);
        return;
    }

    fill(r,0);
    convolve_add(a,b,r);
}

// cube
template< typename T >
inline cube_p<T> convolve( cube<T> const & a,
                           cube<T> const & b)
{
    cube_p<T> r = get_cube<T>(vec3i::one + size(a) - size(b));
    convolve(a,b,*r);
    return r;
}

// qube
template< typename T >
inline qube_p<T> convolve( qube<T> const & a,
                           cube<T> const & b)
{
    size_t am = a.shape()[0];
    size_t ax = a.shape()[1];
    size_t ay = a.shape()[2];
    size_t az = a.shape()[3];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rm = am;
    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    cube_p<T> r = get_qube<T>(rm,rx,ry,rz);
    convolve(a,b,*r);
    return r;
}

// cube
template< typename T >
inline cube_p<T> convolve( ccube_p<T> const & a,
                           ccube_p<T> const & b)
{
    return convolve(*a, *b);
}

// qube
template< typename T >
inline qube_p<T> convolve( qcube_p<T> const & a,
                           ccube_p<T> const & b)
{
    return convolve(*a, *b);
}


// cube
template< typename T >
inline void convolve_flipped_add( cube<T> const & a,
                                  cube<T> const & b,
                                  cube<T> & r) noexcept
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

// qube
template< typename T >
inline void convolve_flipped_add( qube<T> const & a,
                                  cube<T> const & b,
                                  qube<T> & r) noexcept
{
    size_t am = a.shape()[0];
    size_t ax = a.shape()[1];
    size_t ay = a.shape()[2];
    size_t az = a.shape()[3];

    // TODO(lee): convolve_constant_filpped for qube
    // if ( vec3i(ax,ay,az) == size(b) )
    // {
    //     ZI_ASSERT(r.num_elements()==am);
    //     for ( size_t m = 0; m < rm; ++m )
    //         r.data()[0] += convolve_constant_flipped(a,b);
    //     return;
    // }

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rm = am;
    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    ZI_ASSERT(r.shape()[0]==rm);
    ZI_ASSERT(r.shape()[1]==rx);
    ZI_ASSERT(r.shape()[2]==ry);
    ZI_ASSERT(r.shape()[3]==rz);

    for ( size_t m = 0; m < rm; ++m )
        for ( size_t x = 0; x < rx; ++x )
            for ( size_t y = 0; y < ry; ++y )
                for ( size_t z = 0; z < rz; ++z )
                    for ( size_t dx = 0; dx < bx; ++dx )
                        for ( size_t dy = 0; dy < by; ++dy )
                            for ( size_t dz = 0; dz < bz; ++dz )
                                r[m][x][y][z] +=
                                    a[m][ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                    b[bx-1-dx][by-1-dy][bz-1-dz];
}

// cube
template< typename T >
inline void convolve_flipped( cube<T> const & a,
                              cube<T> const & b,
                              cube<T> & r) noexcept
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

// qube
template< typename T >
inline void convolve_flipped( qube<T> const & a,
                              cube<T> const & b,
                              qube<T> & r) noexcept
{
    // TODO(lee): convolve_constant_flipped for qube
    // if ( size(a) == size(b) )
    // {
    //     ZI_ASSERT(r.num_elements()==1);
    //     r.data()[0] = convolve_constant_flipped(a,b);
    //     return;
    // }

    fill(r,0);
    convolve_flipped_add(a,b,r);
}

// cube
template< typename T >
inline cube_p<T> convolve_flipped( cube<T> const & a,
                                   cube<T> const & b)
{
    cube_p<T> r = get_cube<T>(vec3i::one + size(a) - size(b));
    convolve_flipped(a,b,*r);
    return r;
}

// qube
template< typename T >
inline qube_p<T> convolve_flipped( qube<T> const & a,
                                   cube<T> const & b)
{
    size_t am = a.shape()[0];
    size_t ax = a.shape()[1];
    size_t ay = a.shape()[2];
    size_t az = a.shape()[3];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rm = am;
    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    qube_p<T> r = get_qube<T>(rm,rx,ry,rz);
    convolve_flipped(a,b,*r);
    return r;
}

// cube
template< typename T >
inline cube_p<T> convolve_flipped( ccube_p<T> const & a,
                                   ccube_p<T> const & b)
{
    return convolve_flipped(*a, *b);
}

// qube
template< typename T >
inline qube_p<T> convolve_flipped( cqube_p<T> const & a,
                                   ccube_p<T> const & b)
{
    return convolve_flipped(*a, *b);
}


// cube
template< typename T >
inline void convolve_inverse_add( cube<T> const & a,
                                  cube<T> const & b,
                                  cube<T> & r) noexcept
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

// qube
template< typename T >
inline void convolve_inverse_add( qube<T> const & a,
                                  cube<T> const & b,
                                  qube<T> & r) noexcept
{
    // TODO(lee): convolve_constant_inverse_add for qube
    if ( size(b) == vec3i::one )
    {
        convolve_constant_inverse_add(a,b.data()[0],r);
        return;
    }

    size_t am = a.shape()[0];
    size_t ax = a.shape()[1];
    size_t ay = a.shape()[2];
    size_t az = a.shape()[3];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    ZI_ASSERT(r.shape()[0]==am);
    ZI_ASSERT(r.shape()[1]==ax + bx - 1);
    ZI_ASSERT(r.shape()[2]==ay + by - 1);
    ZI_ASSERT(r.shape()[3]==az + bz - 1);

    for ( size_t m = 0; m < rm; ++m )
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
                                r[m][x+fx][y+fy][z+fz] += a[m][x][y][z] * b[dx][dy][dz];
                            }
                }
}

// cube
template< typename T >
inline void convolve_inverse( cube<T> const & a,
                              cube<T> const & b,
                              cube<T> & r) noexcept
{
    if ( size(b) == vec3i::one )
    {
        convolve_constant_inverse(a,b.data()[0],r);
        return;
    }

    fill(r,0);
    convolve_inverse_add(a,b,r);
}

// qube
template< typename T >
inline void convolve_inverse( qube<T> const & a,
                              cube<T> const & b,
                              qube<T> & r) noexcept
{
    // TODO(lee): convolve_constant_inverse for qube
    if ( size(b) == vec3i::one )
    {
        convolve_constant_inverse(a,b.data()[0],r);
        return;
    }

    fill(r,0);
    convolve_inverse_add(a,b,r);
}

// cube
template< typename T >
inline cube_p<T> convolve_inverse( cube<T> const & a,
                                   cube<T> const & b)
{
    cube_p<T> r = get_cube<T>(size(a) + size(b) - vec3i::one);
    convolve_inverse(a,b,*r);
    return r;
}

// qube
template< typename T >
inline qube_p<T> convolve_inverse( qube<T> const & a,
                                   cube<T> const & b)
{
    size_t am = a.shape()[0];
    size_t ax = a.shape()[1];
    size_t ay = a.shape()[2];
    size_t az = a.shape()[3];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rm = am;
    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    qube_p<T> r = get_qube<T>(rm,rx,ry,rz);
    convolve_inverse(a,b,*r);
    return r;
}

// cube
template< typename T >
inline cube_p<T> convolve_inverse( ccube_p<T> const & a,
                                   ccube_p<T> const & b)
{
    return convolve_inverse(*a, *b);
}

// qube
template< typename T >
inline qube_p<T> convolve_inverse( cqube_p<T> const & a,
                                   ccube_p<T> const & b)
{
    return convolve_inverse(*a, *b);
}


}} // namespace znn::v4

#endif
