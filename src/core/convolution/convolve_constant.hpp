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

#ifndef ZNN_CORE_CONVOLUTION_CONVOLVE_CONSTANT_HPP_INCLUDED
#define ZNN_CORE_CONVOLUTION_CONVOLVE_CONSTANT_HPP_INCLUDED

#include "../types.hpp"
#include "../utils.hpp"
#include "../volume_pool.hpp"

namespace zi {
namespace znn {

template< typename T >
inline void convolve_constant_add( const vol<T>& a,
                                   typename identity<T>::type b,
                                   vol<T>& r) noexcept
{
    ZI_ASSERT(size(a)==size(r));

    const T* ap = a.data();
    T* rp = r.data();

    for ( size_t i = 0; i < r.num_elements(); ++i )
        rp[i] += ap[i] * b;
}


template< typename T >
inline void convolve_constant( const vol<T>& a,
                               typename identity<T>::type b,
                               vol<T>& r) noexcept
{
    ZI_ASSERT(size(a)==size(r));

    const T* ap = a.data();
    T* rp = r.data();

    for ( size_t i = 0; i < r.num_elements(); ++i )
        rp[i] = ap[i] * b;
}

template< typename T >
inline vol_p<T> convolve_constant( const vol<T>& a,
                                   typename identity<T>::type b)
{
    vol_p<T> r = get_volume<T>(size(a));
    convolve_constant(a,b,*r);
    return r;
}


template< typename T >
inline vol_p<T> convolve_constant( const vol_p<T>& a,
                                   typename identity<T>::type b)
{
    return convolve_constant(*a,b);
}



template< typename T >
inline T convolve_constant_flipped( const vol<T>& a, const vol<T>& b ) noexcept
{
    ZI_ASSERT(size(a)==size(b));

    T r = 0;

    const T* ap = a.data();
    const T* bp = b.data();

    for ( size_t i = 0; i < a.num_elements(); ++i )
        r += ap[i] * bp[i];

    return r;
}


template<typename T>
inline void convolve_constant_inverse_add( const vol<T>& a,
                                           typename identity<T>::type b,
                                           vol<T>& r)
{
    convolve_constant_add(a,b,r);
}

template<typename T>
inline void convolve_constant_inverse( const vol<T>& a,
                                       typename identity<T>::type b,
                                       vol<T>& r)
{
    convolve_constant(a,b,r);
}

template<typename T>
inline vol_p<T> convolve_constant_inverse( const vol<T>& a,
                                           typename identity<T>::type b )
{
    return convolve_constant(a,b);
}

template<typename T>
inline vol_p<T> convolve_constant_inverse( const vol_p<T>& a,
                                           typename identity<T>::type b )
{
    return convolve_constant(a,b);
}







inline double bf_conv_flipped_constant(const double3d_ptr& ap,
                                       const double3d_ptr& bp)
{
    ASSERT_SAME_SIZE(ap,bp);

    size_t n = ap->num_elements();

    double r = 0;

    double3d& a = *ap;
    double3d& b = *bp;

    for ( size_t i = 0; i < n; ++i )
    {
        r += a.data()[i] * b.data()[i];
    }

    return r;
}

inline double3d_ptr bf_conv_inverse_constant(const double3d_ptr& ap,
                                                   double b)
{
    double3d& a = *ap;

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    double3d_ptr rp = volume_pool.get_double3d(ax,ay,az);
    double3d& r = *rp;

    size_t n = ap->num_elements();

    for ( size_t i = 0; i < n; ++i )
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

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for ( size_t x = 0; x < rx; ++x )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;

                for ( size_t dx = x, wx = bx-1; dx < bx + x; dx += s[0], wx -= s[0] )
                    for ( size_t dy = y, wy = by-1; dy < by + y; dy += s[1], wy -= s[1] )
                        for ( size_t dz = z, wz = bz-1; dz < bz + z; dz += s[2], wz -= s[2] )
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

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for ( size_t x = 0; x < rx; x += s[0])
        for ( size_t y = 0; y < ry; y += s[1] )
            for ( size_t z = 0; z < rz; z += s[2] )
            {
                r[x][y][z] = 0;
                for ( size_t dx = 0; dx < bx; ++dx )
                    for ( size_t dy = 0; dy < by; ++dy )
                        for ( size_t dz = 0; dz < bz; ++dz )
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

    size_t ax = a.shape()[0];
    size_t ay = a.shape()[1];
    size_t az = a.shape()[2];

    size_t bx = b.shape()[0];
    size_t by = b.shape()[1];
    size_t bz = b.shape()[2];

    size_t rx = ax + bx - 1;
    size_t ry = ay + by - 1;
    size_t rz = az + bz - 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    std::fill_n(r.data(), rx*ry*rz, 0);

    for ( size_t dx = 0; dx < bx; dx += s[0] )
        for ( size_t dy = 0; dy < by; dy += s[1])
            for ( size_t dz = 0; dz < bz; dz += s[2] )
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
    return rp;
}

}} // namespace zi::znn

#endif // ZNN_CORE_CONVOLUTION_CONVOLVE_CONSTANT_HPP_INCLUDED
