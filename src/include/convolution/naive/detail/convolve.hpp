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

#include "../../../cube/cube.hpp"
#include "../../../types.hpp"
#include "../../../meta.hpp"

namespace znn { namespace v4 { namespace detail {


template< typename T >
inline void pure_convolve_add( cube<T> const & a,
                               cube<T> const & b,
                               cube<T> & r) noexcept
{
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
inline void pure_convolve( cube<T> const & a,
                           cube<T> const & b,
                           cube<T> & r) noexcept
{
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
            {
                r[x][y][z] = 0;
                for ( size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
                    for ( size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                        for ( size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                            r[x][y][z] += a[dx][dy][dz] * b[wx][wy][wz];
            }
}

template< typename T >
inline void pure_convolve_flipped( cube<T> const & a,
                                   cube<T> const & b,
                                   cube<T> & r) noexcept
{
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
            {
                r[x][y][z] = 0;
                for ( size_t dx = 0; dx < bx; ++dx )
                    for ( size_t dy = 0; dy < by; ++dy )
                        for ( size_t dz = 0; dz < bz; ++dz )
                            r[x][y][z] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
            }
}


template< typename T >
inline void pure_convolve_inverse_add( cube<T> const & a,
                                       cube<T> const & b,
                                       cube<T> & r) noexcept
{
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
inline void pure_convolve_inverse( cube<T> const & a,
                                   cube<T> const & b,
                                   cube<T> & r) noexcept
{
    fill(r,0);
    pure_convolve_inverse_add(a,b,r);
}

}}} // namespace znn::v4::detail
