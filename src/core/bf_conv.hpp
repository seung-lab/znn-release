//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_BF_CONV_HPP_INCLUDED
#define ZNN_BF_CONV_HPP_INCLUDED

#include "types.hpp"
#include "volume_pool.hpp"

namespace zi {
namespace znn {

inline double3d_ptr bf_conv(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;
    std::size_t rz = az - bz + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
                    for ( std::size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                        for ( std::size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                        {
                            r[x][y][z] +=
                                a[dx][dy][dz] *
                                b[wx][wy][wz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_old(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;
    std::size_t rz = az - bz + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = 0; dx < bx; ++dx )
                    for ( std::size_t dy = 0; dy < by; ++dy )
                        for ( std::size_t dz = 0; dz < bz; ++dz )
                        {
                            r[x][y][z] +=
                                a[x+dx][y+dy][z+dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_flipped(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;
    std::size_t rz = az - bz + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = 0; dx < bx; ++dx )
                    for ( std::size_t dy = 0; dy < by; ++dy )
                        for ( std::size_t dz = 0; dz < bz; ++dz )
                        {
                            r[x][y][z] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_inverse(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax + bx - 1;
    std::size_t ry = ay + by - 1;
    std::size_t rz = az + bz - 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    std::fill_n(r.data(), rx*ry*rz, 0);

    for ( std::size_t dx = 0; dx < bx; ++dx )
        for ( std::size_t dy = 0; dy < by; ++dy )
            for ( std::size_t dz = 0; dz < bz; ++dz )
            {
                std::size_t fx = bx - 1 - dx;
                std::size_t fy = by - 1 - dy;
                std::size_t fz = bz - 1 - dz;

                for ( std::size_t x = 0; x < ax; ++x )
                    for ( std::size_t y = 0; y < ay; ++y )
                        for ( std::size_t z = 0; z < az; ++z )
                        {
                            r[x+fx][y+fy][z+fz] += a[x][y][z] * b[dx][dy][dz];
                        }
            }
    return rp;
}

inline double3d_ptr bf_conv_constant(const double3d_ptr& ap,
                                           double b)
{
    double3d& a = *ap;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    double3d_ptr rp = volume_pool.get_double3d(ax,ay,az);
    double3d& r = *rp;

    for ( std::size_t x = 0; x < ax; ++x )
        for ( std::size_t y = 0; y < ay; ++y )
            for ( std::size_t z = 0; z < az; ++z )
            {
                r[x][y][z] = a[x][y][z] * b;
            }

    return rp;
}

inline double bf_conv_flipped_constant(const double3d_ptr& ap,
                                       const double3d_ptr& bp)
{
    ASSERT_SAME_SIZE(ap,bp);

    std::size_t n = ap->num_elements();

    double r = 0;

    double3d& a = *ap;
    double3d& b = *bp;

    for ( std::size_t i = 0; i < n; ++i )
    {
        r += a.data()[i] * b.data()[i];
    }

    return r;
}

inline double3d_ptr bf_conv_inverse_constant(const double3d_ptr& ap,
                                                   double b)
{
    double3d& a = *ap;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    double3d_ptr rp = volume_pool.get_double3d(ax,ay,az);
    double3d& r = *rp;

    std::size_t n = ap->num_elements();

    for ( std::size_t i = 0; i < n; ++i )
    {
        r.data()[i] = a.data()[i] * b;
    }

    return rp;
}


inline double3d_ptr bf_conv_sparse(const double3d_ptr& ap,
                                   const double3d_ptr& bp,
                                   const vec3i& s)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;
    std::size_t rz = az - bz + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
             
                for ( std::size_t dx = x, wx = bx-1; dx < bx + x; dx += s[0], wx -= s[0] )
                    for ( std::size_t dy = y, wy = by-1; dy < by + y; dy += s[1], wy -= s[1] )
                        for ( std::size_t dz = z, wz = bz-1; dz < bz + z; dz += s[2], wz -= s[2] )
                        {
                            r[x][y][z] +=
                                a[dx][dy][dz] *
                                b[wx][wy][wz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_flipped_sparse(const double3d_ptr& ap,
                                                 const double3d_ptr& bp,
                                                 const vec3i& s)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;
    std::size_t rz = az - bz + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for ( std::size_t x = 0; x < rx; x += s[0])
        for ( std::size_t y = 0; y < ry; y += s[1] )
            for ( std::size_t z = 0; z < rz; z += s[2] )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = 0; dx < bx; ++dx )
                    for ( std::size_t dy = 0; dy < by; ++dy )
                        for ( std::size_t dz = 0; dz < bz; ++dz )
                        {
                            r[x][y][z] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_inverse_sparse(const double3d_ptr& ap,
                                                 const double3d_ptr& bp,
                                                 const vec3i& s)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax + bx - 1;
    std::size_t ry = ay + by - 1;
    std::size_t rz = az + bz - 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    std::fill_n(r.data(), rx*ry*rz, 0);

    for ( std::size_t dx = 0; dx < bx; dx += s[0] )
        for ( std::size_t dy = 0; dy < by; dy += s[1])
            for ( std::size_t dz = 0; dz < bz; dz += s[2] )
            {
                std::size_t fx = bx - 1 - dx;
                std::size_t fy = by - 1 - dy;
                std::size_t fz = bz - 1 - dz;

                for ( std::size_t x = 0; x < ax; ++x )
                    for ( std::size_t y = 0; y < ay; ++y )
                        for ( std::size_t z = 0; z < az; ++z )
                        {
                            r[x+fx][y+fy][z+fz] += a[x][y][z] * b[dx][dy][dz];
                        }
            }
    return rp;
}

}} // namespace zi::znn

#endif // ZNN_BF_CONV_HPP_INCLUDED