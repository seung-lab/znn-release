//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
//                     Kisuk Lee           <kisuklee@mit.edu>
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

#ifndef ZNN_VOLUME_UTILS_HPP_INCLUDED
#define ZNN_VOLUME_UTILS_HPP_INCLUDED

#include "types.hpp"
#include "utils.hpp"
#include "measure.hpp"
#include "volume_pool.hpp"

#include <zi/parallel/numeric.hpp>
#include <zi/parallel/algorithm.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <cstdlib>

namespace zi {
namespace znn {
namespace volume_utils {

template <typename T>
inline vec3i volume_size(const T& a)
{
    return vec3i(a->shape()[0], a->shape()[1], a->shape()[2]);
}

template <typename T>
inline void add_to(const T& a, const T& b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        b->data()[i] += a->data()[i];
    }
}

inline void mul_add_to(double c, double3d_ptr a, double3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        b->data()[i] += c * a->data()[i];
    }
}

inline void elementwise_mul_by(double3d_ptr a, double3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] *= b->data()[i];
    }
}

inline void elementwise_mul_by(complex3d_ptr a, complex3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] *= b->data()[i];
    }
}

inline void elementwise_mul(double3d_ptr r, double3d_ptr a, double3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        r->data()[i] = b->data()[i] * a->data()[i];;
    }
}

inline void elementwise_mul(complex3d_ptr r, complex3d_ptr a, complex3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        r->data()[i] = b->data()[i] * a->data()[i];;
    }
}

inline void elementwise_masking(double3d_ptr a, bool3d_ptr b)
{
    if ( !b ) return;

    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        if( b->data()[i] == false )
        {
            a->data()[i] = static_cast<double>(0);
        }
    }
}

inline void elementwise_mul_by(double3d_ptr a, double b)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] *= b;
    }
}

inline void elementwise_div_by(double3d_ptr a, double b)
{
    PROFILE_FUNCTION();
    double c = static_cast<double>(1) / b;
    elementwise_mul_by(a,c);
}

inline void elementwise_sub_from(double c, double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] -= c;
    }
}

inline void elementwise_abs(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = std::abs(a->data()[i]);
    }
}

inline void elementwise_max(double3d_ptr a, double c)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = std::max(a->data()[i], c);
    }
}

inline complex3d_ptr elementwise_mul(complex3d_ptr a, complex3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    complex3d_ptr r = volume_pool.get_complex3d(a);

    std::size_t n = a->num_elements();

    STRONG_ASSERT(b.use_count()>0);

    for ( std::size_t i = 0; i < n; ++i )
    {
        r->data()[i] = a->data()[i] * b->data()[i];
    }

    return r;
}

inline void elementwise_and(bool3d_ptr a, bool3d_ptr b)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] &= b->data()[i];
    }
}

inline void sub_from_mul(double3d_ptr a, double3d_ptr b, double c)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        b->data()[i] -= a->data()[i];
        b->data()[i] *= c;
    }
}

inline void sub_from_mul(double3d_ptr r, double3d_ptr a, double3d_ptr b,
                         double c)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    ASSERT_SAME_SIZE(a,r);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        r->data()[i] = c*(a->data()[i] - b->data()[i]);
    }
}

inline void sub_from(double3d_ptr a, double3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        b->data()[i] -= a->data()[i];
    }
}

inline double sum_all(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();
    return zi::accumulate(a->data(), a->data()+n, static_cast<double>(0));
}

inline double sum_all(std::list<double3d_ptr> a)
{
    double ret = static_cast<double>(0);
    FOR_EACH( it, a )
    {
        ret += volume_utils::sum_all(*it);
    }
    return ret;
}

inline double nnz(bool3d_ptr a)
{
    PROFILE_FUNCTION();
    double ret = static_cast<double>(0);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        if( a->data()[i] ) ret++;
    }
    return ret;
}

inline double nnz(std::list<bool3d_ptr> a)
{
    PROFILE_FUNCTION();
    double ret = static_cast<double>(0);
    FOR_EACH( it, a )
    {
        ret += nnz(*it);
    }
    return ret;
}

inline double square_sum(double3d_ptr a)
{
    PROFILE_FUNCTION();
    double ret = static_cast<double>(0);
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        ret += a->data()[i]*a->data()[i];
    }
    return ret;
}

inline void fill_indices(long3d_ptr a)
{
    PROFILE_FUNCTION();
    long_t n = a->shape()[0]*a->shape()[1]*a->shape()[2];
    for ( long_t i = 0; i < n; ++i )
    {
        a->data()[i] = i;
    }
}

template <class A>
inline void print(const A& a)
{
    PROFILE_FUNCTION();
    std::size_t rx = a->shape()[0];
    std::size_t ry = a->shape()[1];
    std::size_t rz = a->shape()[2];

    for ( std::size_t x = 0; x < rx; ++x )
    {
        if ( x > 0 )
        {
            std::cout << "\n\n";
        }
        for ( std::size_t y = 0; y < ry; ++y )
        {
            if ( y > 0 )
            {
                std::cout << "\n";
            }
            for ( std::size_t z = 0; z < rz; ++z )
            {
                if ( z > 0 )
                {
                    std::cout << ' ';
                }
                std::cout << (*a)[x][y][z];
            }
        }
    }
}

template <class A>
inline void print_in_matlab_format(const A& a)
{
    PROFILE_FUNCTION();
    std::size_t rx = a->shape()[0];
    std::size_t ry = a->shape()[1];
    std::size_t rz = a->shape()[2];

    for ( std::size_t z = 0; z < rz; ++z )
    {
        if ( z > 0 )
        {
            std::cout << "\n\n";
        }
        for ( std::size_t x = 0; x < rx; ++x )
        {
            if ( x > 0 )
            {
                std::cout << "\n";
            }
            for ( std::size_t y = 0; y < ry; ++y )
            {
                if ( y > 0 )
                {
                    std::cout << ' ';
                }
                std::cout << (*a)[x][y][z];
            }
        }
    }
    std::cout << "\n\n";
}

template< class V >
inline void zero_out(const V& a)
{
    PROFILE_FUNCTION();
    std::fill_n(a->data(), a->shape()[0]*a->shape()[1]*a->shape()[2], 0);
}

template< class V >
inline void fill_one(const V& a)
{
    PROFILE_FUNCTION();
    std::fill_n(a->data(), a->shape()[0]*a->shape()[1]*a->shape()[2], 1);
}

template< class V, class T >
inline void fill_n(const V& a, T val)
{
    PROFILE_FUNCTION();
    std::fill_n(a->data(), a->shape()[0]*a->shape()[1]*a->shape()[2], val);
}

inline void distribute_values(double3d_ptr vals, long3d_ptr idxs, double3d_ptr out)
{
    PROFILE_FUNCTION();
    zero_out(out);
    std::size_t n = vals->shape()[0]*vals->shape()[1]*vals->shape()[2];

    ZI_ASSERT(n==idxs->shape()[0]*idxs->shape()[1]*idxs->shape()[2]);

    double* val = vals->data();
    long_t* idx = idxs->data();
    double* val_end = val + n;

    for ( ; val < val_end; ++val, ++idx)
    {
        out->data()[*idx] = *val;
    }
}

inline double3d_ptr rand_volume(std::size_t x, std::size_t y, std::size_t z,
                                double min, double max)
{
    PROFILE_FUNCTION();
    double3d_ptr a = volume_pool.get_double3d(x,y,z);

    std::size_t n = a->num_elements();

    double range = max-min;

    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = range * ( static_cast<double>(std::rand()%1000000) / 999999 ) + min;
    }

    return a;
}

inline double3d_ptr crop(double3d_ptr a, std::size_t x, std::size_t y, std::size_t z)
{
    PROFILE_FUNCTION();
    double3d_ptr r = volume_pool.get_double3d(x,y,z);
    *r = (*a)[boost::indices[range(0,x)][range(0,y)][range(0,z)]];

    return r;
}

inline bool3d_ptr crop(bool3d_ptr a, std::size_t x, std::size_t y, std::size_t z)
{
    PROFILE_FUNCTION();
    bool3d_ptr r = volume_pool.get_bool3d(x,y,z);
    *r = (*a)[boost::indices[range(0,x)][range(0,y)][range(0,z)]];

    return r;
}

inline long3d_ptr crop(long3d_ptr a, std::size_t x, std::size_t y, std::size_t z)
{
    PROFILE_FUNCTION();
    long3d_ptr r = volume_pool.get_long3d(x,y,z);
    *r = (*a)[boost::indices[range(0,x)][range(0,y)][range(0,z)]];

    return r;
}

inline double3d_ptr crop(double3d_ptr a,
                         std::size_t ox, std::size_t oy, std::size_t oz,
                         std::size_t x, std::size_t y, std::size_t z)
{
    PROFILE_FUNCTION();
    double3d_ptr r = volume_pool.get_double3d(x,y,z);
    *r = (*a)[boost::indices[range(ox,x+ox)][range(oy,y+oy)][range(oz,z+oz)]];

    return r;
}

inline bool3d_ptr crop(bool3d_ptr a,
                       std::size_t ox, std::size_t oy, std::size_t oz,
                       std::size_t x, std::size_t y, std::size_t z)
{
    PROFILE_FUNCTION();
    bool3d_ptr r = volume_pool.get_bool3d(x,y,z);
    *r = (*a)[boost::indices[range(ox,x+ox)][range(oy,y+oy)][range(oz,z+oz)]];

    return r;
}

inline long3d_ptr crop(long3d_ptr a,
                       std::size_t ox, std::size_t oy, std::size_t oz,
                       std::size_t x, std::size_t y, std::size_t z)
{
    PROFILE_FUNCTION();
    long3d_ptr r = volume_pool.get_long3d(x,y,z);
    *r = (*a)[boost::indices[range(ox,x+ox)][range(oy,y+oy)][range(oz,z+oz)]];

    return r;
}

inline double3d_ptr crop(double3d_ptr a, const vec3i& off, const vec3i& sz )
{
    return crop(a, off[0], off[1], off[2], sz[0], sz[1], sz[2] );
}

inline long3d_ptr crop(long3d_ptr a, const vec3i& off, const vec3i& sz )
{
    return crop(a, off[0], off[1], off[2], sz[0], sz[1], sz[2] );
}

inline bool3d_ptr crop(bool3d_ptr a, const vec3i& off, const vec3i& sz )
{
    return crop(a, off[0], off[1], off[2], sz[0], sz[1], sz[2] );
}

inline double3d_ptr crop_right(const double3d_ptr& a, const vec3i& s)
{
    PROFILE_FUNCTION();
    vec3i off = size_of(a) - s;
    return crop(a, off[0], off[1], off[2], s[0], s[1], s[2]);
}

inline double3d_ptr crop_left(const double3d_ptr& a, const vec3i& s)
{
    return crop(a, 0, 0, 0, s[0], s[1], s[2]);
}

inline double3d_ptr zero_pad(double3d_ptr a, std::size_t x, std::size_t y, std::size_t z)
{
    PROFILE_FUNCTION();
    std::size_t ox = a->shape()[0];
    std::size_t oy = a->shape()[1];
    std::size_t oz = a->shape()[2];

    double3d_ptr r = volume_pool.get_double3d(x,y,z);

    zero_out(r);

    (*r)[boost::indices[range(0,ox)][range(0,oy)][range(0,oz)]]
        = (*a);

    return r;
}

inline double3d_ptr zero_pad(double3d_ptr a, vec3i d)
{
    return zero_pad(a,d[0],d[1],d[2]);
}

inline void zero_pad(double3d_ptr r, double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t ox = a->shape()[0];
    std::size_t oy = a->shape()[1];
    std::size_t oz = a->shape()[2];

    zero_out(r);

    (*r)[boost::indices[range(0,ox)][range(0,oy)][range(0,oz)]]
        = (*a);

}

inline double3d_ptr flip(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    double3d_ptr r = volume_pool.get_double3d(x,y,z);

    for ( std::size_t i = 0; i < x; ++i )
    {
        for ( std::size_t j = 0; j < y; ++j )
        {
            for ( std::size_t k = 0; k < z; ++k )
            {
                (*r)[x-i-1][y-j-1][z-k-1] = (*a)[i][j][k];
            }
        }
    }

    return r;
}

inline complex3d_ptr flip(complex3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    complex3d_ptr r = volume_pool.get_complex3d(a);

    for ( std::size_t i = 0; i < x; ++i )
    {
        for ( std::size_t j = 0; j < y; ++j )
        {
            for ( std::size_t k = 0; k < z; ++k )
            {
                (*r)[x-i-1][y-j-1][z-k-1] = std::conj((*a)[i][j][k]);
            }
        }
    }

    return r;
}

inline bool3d_ptr flip(bool3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    bool3d_ptr r = volume_pool.get_bool3d(x,y,z);

    for ( std::size_t i = 0; i < x; ++i )
    {
        for ( std::size_t j = 0; j < y; ++j )
        {
            for ( std::size_t k = 0; k < z; ++k )
            {
                (*r)[x-i-1][y-j-1][z-k-1] = (*a)[i][j][k];
            }
        }
    }

    return r;
}

// test implementation - should be elaborated later
// dim: binary coding
inline double3d_ptr flipdim(double3d_ptr a, std::size_t dim)
{
    dim = dim % 8;
    if ( dim == 0  )
    {
        return a;
    }

    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    double3d_ptr r = volume_pool.get_double3d(x,y,z);

    if ( dim == 4 ) // 100 x:1 y:0 z:0
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[x-i-1][j][k] = (*a)[i][j][k];
    }
    else if( dim == 2 ) // 010 x:0 y:1 z:0
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[i][y-j-1][k] = (*a)[i][j][k];
    }
    else if ( dim == 1 ) // 001 x:0 y:0 z:1
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[i][j][z-k-1] = (*a)[i][j][k];
    }
    else if ( dim == 6 ) // 110 x:1 y:1 z:0
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[x-i-1][y-j-1][k] = (*a)[i][j][k];
    }
    else if ( dim == 5 ) // 101 x:1 y:0 z:1
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[x-i-1][j][z-k-1] = (*a)[i][j][k];
    }
    else if ( dim == 3 ) // 011 x:0 y:1 z:1
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[i][y-j-1][z-k-1] = (*a)[i][j][k];
    }
    else if ( dim == 7 ) // 111 x:1 y:1 z:1
    {
        r = flip(a);
    }

    return r;
}

inline void flipdim(std::list<double3d_ptr>& vl, std::size_t dim)
{
    PROFILE_FUNCTION();
    FOR_EACH( it, vl )
    {
        (*it) = flipdim(*it, dim);
    }
}

// test implementation - should be elaborated later
// dim: binary coding
inline bool3d_ptr flipdim(bool3d_ptr a, std::size_t dim)
{
    dim = dim % 8;
    if ( dim == 0 )
    {
        return a;
    }

    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    bool3d_ptr r = volume_pool.get_bool3d(x,y,z);

    if ( dim == 4 ) // 100 x:1 y:0 z:0
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[x-i-1][j][k] = (*a)[i][j][k];
    }
    else if( dim == 2 ) // 010 x:0 y:1 z:0
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[i][y-j-1][k] = (*a)[i][j][k];
    }
    else if ( dim == 1 ) // 001 x:0 y:0 z:1
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[i][j][z-k-1] = (*a)[i][j][k];
    }
    else if ( dim == 6 ) // 110 x:1 y:1 z:0
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[x-i-1][y-j-1][k] = (*a)[i][j][k];
    }
    else if ( dim == 5 ) // 101 x:1 y:0 z:1
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[x-i-1][j][z-k-1] = (*a)[i][j][k];
    }
    else if ( dim == 3 ) // 011 x:0 y:1 z:1
    {
        for ( std::size_t i = 0; i < x; ++i )
            for ( std::size_t j = 0; j < y; ++j )
                for ( std::size_t k = 0; k < z; ++k )
                    (*r)[i][y-j-1][z-k-1] = (*a)[i][j][k];
    }
    else if ( dim == 7 ) // 111 x:1 y:1 z:1
    {
        r = flip(a);
    }

    return r;
}

inline void flipdim(std::list<bool3d_ptr>& vl, std::size_t dim)
{
    PROFILE_FUNCTION();
    FOR_EACH( it, vl )
    {
        (*it) = flipdim(*it, dim);
    }
}

inline double3d_ptr transpose(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    double3d_ptr r = volume_pool.get_double3d(x,y,z);

    for ( std::size_t i = 0; i < x; ++i )
    {
        for ( std::size_t j = 0; j < y; ++j )
        {
            for ( std::size_t k = 0; k < z; ++k )
            {
                (*r)[j][i][k] = (*a)[i][j][k];
            }
        }
    }

    return r;
}

inline void transpose(std::list<double3d_ptr>& vl)
{
    PROFILE_FUNCTION();
    FOR_EACH( it, vl )
    {
        (*it) = transpose(*it);
    }
}

inline bool3d_ptr transpose(bool3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    bool3d_ptr r = volume_pool.get_bool3d(x,y,z);

    for ( std::size_t i = 0; i < x; ++i )
    {
        for ( std::size_t j = 0; j < y; ++j )
        {
            for ( std::size_t k = 0; k < z; ++k )
            {
                (*r)[j][i][k] = (*a)[i][j][k];
            }
        }
    }

    return r;
}

inline void transpose(std::list<bool3d_ptr>& vl)
{
    PROFILE_FUNCTION();
    FOR_EACH( it, vl )
    {
        (*it) = volume_utils::transpose(*it);
    }
}

inline void normalize(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();
    elementwise_div_by(a,static_cast<double>(n));
}


inline double3d_ptr normalize_flip(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    long double n = x * y * z;
    double c = static_cast<long double>(1)/n;

    double3d_ptr r = volume_pool.get_double3d(x,y,z);

    for ( std::size_t i = 0; i < x; ++i )
    {
        for ( std::size_t j = 0; j < y; ++j )
        {
            for ( std::size_t k = 0; k < z; ++k )
            {
                (*r)[x-i-1][y-j-1][z-k-1] = (*a)[i][j][k] * c;
            }
        }
    }

    return r;
}

inline double3d_ptr zero_out_nongrid(double3d_ptr a, const vec3i& s)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    double3d_ptr r = volume_pool.get_double3d(x,y,z);
    zero_out(r);

    for ( std::size_t i = 0; i < x; i+=s[0] )
    {
        for ( std::size_t j = 0; j < y; j+=s[1] )
        {
            for ( std::size_t k = 0; k < z; k+=s[2] )
            {
                (*r)[i][j][k] = (*a)[i][j][k];
            }
        }
    }

    return r;
}

inline double3d_ptr sparse_compress(double3d_ptr a, const vec3i& s)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    std::size_t rx = (s[0] == 1 ? x : x/s[0]+1);
    std::size_t ry = (s[1] == 1 ? y : y/s[1]+1);
    std::size_t rz = (s[2] == 1 ? z : z/s[2]+1);

    double3d_ptr r = volume_pool.get_double3d(rx,ry,rz);

    for ( std::size_t i = 0, ii = 0; i < x; i+=s[0], ++ii )
    {
        for ( std::size_t j = 0, jj = 0; j < y; j+=s[1], ++jj )
        {
            for ( std::size_t k = 0, kk = 0; k < z; k+=s[2], ++kk )
            {
                (*r)[ii][jj][kk] = (*a)[i][j][k];
            }
        }
    }

    return r;
}

inline bool3d_ptr sparse_compress(bool3d_ptr a, const vec3i& s)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];
    
    std::size_t rx = (s[0] == 1 ? x : x/s[0]+1);
    std::size_t ry = (s[1] == 1 ? y : y/s[1]+1);
    std::size_t rz = (s[2] == 1 ? z : z/s[2]+1);

    bool3d_ptr r = volume_pool.get_bool3d(rx,ry,rz);

    for ( std::size_t i = 0, ii = 0; i < x; i+=s[0], ++ii )
    {
        for ( std::size_t j = 0, jj = 0; j < y; j+=s[1], ++jj )
        {
            for ( std::size_t k = 0, kk = 0; k < z; k+=s[2], ++kk )
            {
                (*r)[ii][jj][kk] = (*a)[i][j][k];
            }
        }
    }

    return r;
}

inline double3d_ptr sparse_decompress(double3d_ptr a, const vec3i& s)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    double3d_ptr r = volume_pool.get_double3d((x-1)*s[0]+1,
                                              (y-1)*s[1]+1,
                                              (z-1)*s[2]+1);
    zero_out(r);

    for ( std::size_t i = 0, ii = 0; ii < x; i+=s[0], ++ii )
    {
        for ( std::size_t j = 0, jj = 0; jj < y; j+=s[1], ++jj )
        {
            for ( std::size_t k = 0, kk = 0; kk < z; k+=s[2], ++kk )
            {
                (*r)[i][j][k] = (*a)[ii][jj][kk];
            }
        }
    }

    return r;
}

inline bool3d_ptr sparse_decompress(bool3d_ptr a, const vec3i& s)
{
    PROFILE_FUNCTION();
    std::size_t x = a->shape()[0];
    std::size_t y = a->shape()[1];
    std::size_t z = a->shape()[2];

    bool3d_ptr r = volume_pool.get_bool3d((x-1)*s[0]+1,
                                          (y-1)*s[1]+1,
                                          (z-1)*s[2]+1);
    zero_out(r);

    for ( std::size_t i = 0, ii = 0; ii < x; i+=s[0], ++ii )
    {
        for ( std::size_t j = 0, jj = 0; jj < y; j+=s[1], ++jj )
        {
            for ( std::size_t k = 0, kk = 0; kk < z; k+=s[2], ++kk )
            {
                (*r)[i][j][k] = (*a)[ii][jj][kk];
            }
        }
    }

    return r;
}

// volume file input/output
template <typename T>
inline bool load( T a, const std::string& fname )
{
    vec3i sz = size_of(a);
    std::size_t elemsz = sizeof(a->data()[0]);

    // open file
    FILE* fvol = fopen(fname.c_str(), "r");
    if ( !fvol ) return false;

    // load each label
    for ( std::size_t z = 0; z < sz[2]; ++z )
        for ( std::size_t y = 0; y < sz[1]; ++y )
            for ( std::size_t x = 0; x < sz[0]; ++x )
                static_cast<void>(fread(&((*a)[x][y][z]), elemsz, 1, fvol));

    return true;
}

template <typename T>
inline void save( T a, const std::string& fname )
{
    std::ofstream fvol(fname.c_str(), (std::ios::out | std::ios::binary) );

    vec3i sz = size_of(a);
    std::size_t elemsz = sizeof(a->data()[0]);

    for ( std::size_t z = 0; z < sz[2]; ++z )
        for ( std::size_t y = 0; y < sz[1]; ++y )
            for ( std::size_t x = 0; x < sz[0]; ++x )
                fvol.write( reinterpret_cast<char*>(&((*a)[x][y][z])), elemsz );
}

template <typename T>
inline void save_append( T a, const std::string& fname )
{
    std::ofstream fvol(fname.c_str(), (std::ios::out | std::ios::binary | std::ios::app ) );

    vec3i sz = size_of(a);
    std::size_t elemsz = sizeof(a->data()[0]);

    for ( std::size_t z = 0; z < sz[2]; ++z )
        for ( std::size_t y = 0; y < sz[1]; ++y )
            for ( std::size_t x = 0; x < sz[0]; ++x )
                fvol.write( reinterpret_cast<char*>(&((*a)[x][y][z])), elemsz );
}

template <typename T>
inline void load_list( std::list<T> a, const std::string& fname )
{
    std::size_t idx = 0;
    FOR_EACH( it, a )
    {
        std::string idx_str = boost::lexical_cast<std::string>(idx++);
        volume_utils::load(*it, fname + "." + idx_str);
    }
}

template <typename T>
inline void save_list( std::list<T> a, const std::string& fname )
{
    std::size_t idx = 0;
    FOR_EACH( it, a )
    {
        std::string idx_str = boost::lexical_cast<std::string>(idx++);
        volume_utils::save(*it, fname + "." + idx_str);
        export_size_info(size_of(*it), fname);
    }
}

inline void add_list_to( std::list<double3d_ptr> a, 
                         std::list<double3d_ptr> b )
{
    std::list<double3d_ptr>::iterator at = a.begin();
    FOR_EACH( bt, b )
    {
        volume_utils::add_to(*at++,*bt);
    }
}

// lbl is assumed to be either min or max values
inline double3d_ptr classification_error(double3d_ptr prob, double3d_ptr lbl, double thresh = 0.5)
{
    ASSERT_SAME_SIZE(prob,lbl);
    vec3i sz = size_of(prob);
    std::size_t n = lbl->num_elements();
    
    double3d_ptr ret = volume_pool.get_double3d(sz);
    for ( std::size_t i = 0; i < n; ++i )
    {
        double pred = (prob->data()[i] > thresh ? 
                        static_cast<double>(1):static_cast<double>(-1));
        double truth = (lbl->data()[i] > thresh ?
                        static_cast<double>(1):static_cast<double>(-1));
        
        // binary classification
        ret->data()[i] = 
            ((pred * truth > static_cast<double>(0)) ?
                static_cast<double>(0):static_cast<double>(1));
    }
    return ret;
}

// generic softmax
inline std::list<double3d_ptr> softmax(std::list<double3d_ptr> vl)
{
    STRONG_ASSERT( !vl.empty() );
    FOR_EACH( it, vl )
    {
        ASSERT_SAME_SIZE(vl.front(),*it);
    }

    vec3i sz = volume_size(vl.front());
    std::size_t n = sz[0]*sz[1]*sz[2];

    // normalization
    double3d_ptr sum = volume_pool.get_double3d(sz);
    zero_out(sum);
    FOR_EACH( it, vl )
    {
        for ( std::size_t i = 0; i < n; ++i )
        {
            sum->data()[i] += std::exp((*it)->data()[i]);
        }
    }

    // activation
    std::list<double3d_ptr> ret;
    FOR_EACH( it, vl )
    {
        double3d_ptr act = volume_pool.get_double3d(sz);
        for ( std::size_t i = 0; i < n; ++i )
        {
            act->data()[i] = std::exp((*it)->data()[i]) / sum->data()[i];
        }
        ret.push_back(act);
    }
    return ret;
}

// for multinomial cross-entropy
inline double3d_ptr cross_entropy( double3d_ptr v, double3d_ptr l )
{
    ASSERT_SAME_SIZE(v,l);
    
    double3d_ptr ret = volume_pool.get_double3d(v);
    
    std::size_t n = v->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        ret->data()[i] = -(l->data()[i]*std::log(v->data()[i]));
    }

    return ret;
}

// multinomial cross-entropy
inline double3d_ptr cross_entropy(std::list<double3d_ptr> vl, std::list<double3d_ptr> ll)
{
    STRONG_ASSERT(!vl.empty());
    STRONG_ASSERT(vl.size() == ll.size());

    double3d_ptr ret = volume_pool.get_double3d(vl.front());
    volume_utils::zero_out(ret);

    std::list<double3d_ptr>::iterator lit = ll.begin();
    FOR_EACH( it, vl )
    {
        volume_utils::add_to(cross_entropy(*it,*lit++),ret);
    }

    return ret;
}

// binomial cross-entropy
inline double3d_ptr binomial_cross_entropy( double3d_ptr v, double3d_ptr l )
{
    ASSERT_SAME_SIZE(v,l);

    double3d_ptr ret = volume_pool.get_double3d(v);
    
    std::size_t n = v->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        ret->data()[i] = -(l->data()[i]*std::log(v->data()[i]))
                         -(1 - l->data()[i])*std::log(1 - v->data()[i]);
    }

    return ret;
}

// binomial cross-entropy
// each output node is considered to be an individudal binomial unit
inline std::list<double3d_ptr> binomial_cross_entropy( std::list<double3d_ptr> vl, 
                                                       std::list<double3d_ptr> ll )
{
    STRONG_ASSERT(!vl.empty());
    STRONG_ASSERT(vl.size() == ll.size());

    std::list<double3d_ptr> ret;

    std::list<double3d_ptr>::iterator lit = ll.begin();
    FOR_EACH( it, vl )
    {
        ret.push_back(binomial_cross_entropy(*it,*lit++));
    }

    return ret;
}

// for input normalization
inline double get_mean(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();

    return volume_utils::sum_all(a)/static_cast<double>(n);
}

inline double get_std(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();

    double mean = volume_utils::get_mean(a);
    double3d_ptr b = volume_pool.get_double3d(a);
    for ( std::size_t i = 0; i < n; ++i )
    {
        b->data()[i] = (a->data()[i] - mean)*(a->data()[i] - mean);
    }
    double var = volume_utils::get_mean(b);
    return std::sqrt(var);
}

inline void normalize_volume(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();

    double mean = volume_utils::get_mean(a);
    double stdev = volume_utils::get_std(a);
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = (a->data()[i] - mean)/stdev;
    }
}

inline void normalize_volume_for_2D(double3d_ptr a)
{
    PROFILE_FUNCTION();
    vec3i volsz = size_of(a);

    // 2D slice size
    std::size_t sx = volsz[0];
    std::size_t sy = volsz[1];
    std::size_t sz = 1;

    // offset
    std::size_t ox = 0;
    std::size_t oy = 0;

    for ( std::size_t oz = 0; oz < volsz[2]; ++oz )
    {
        double3d_ptr slice = volume_utils::crop(a,ox,oy,oz,sx,sy,sz);
        volume_utils::normalize_volume(slice);

        (*a)[boost::indices[range(ox,ox+sx)][range(oy,oy+sy)][range(oz,oz+sz)]] =
                (*slice)[boost::indices[range(0,sx)][range(0,sy)][range(0,sz)]];
    }
}

inline void unit_transform(double3d_ptr a)
{
    std::size_t n = a->num_elements();
    double min_val = *std::min_element(a->origin(), a->origin() + n);
    double max_val = *std::max_element(a->origin(), a->origin() + n);
    double range = max_val - min_val;

    // std::cout << "Transform from [" << min_val << "," << max_val << "] ";
    
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = (a->data()[i] - min_val)/range;
    }

    min_val = *std::min_element(a->origin(), a->origin() + n);
    max_val = *std::max_element(a->origin(), a->origin() + n);

    // std::cout << "to [" << min_val << "," << max_val << "]" << std::endl;
}

inline void transform(double3d_ptr a, double upper_bound, double lower_bound)
{
    STRONG_ASSERT(upper_bound > lower_bound);

    std::size_t n = a->num_elements();
    double min_val = *std::min_element(a->origin(), a->origin() + n);
    double max_val = *std::max_element(a->origin(), a->origin() + n);
    double old_range = max_val - min_val;
    double new_range = upper_bound - lower_bound;

    std::cout << "Transform from [" << min_val << "," << max_val << "] ";
    
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = new_range*((a->data()[i] - min_val)/old_range) + lower_bound;
    }

    min_val = *std::min_element(a->origin(), a->origin() + n);
    max_val = *std::max_element(a->origin(), a->origin() + n);

    std::cout << "to [" << min_val << "," << max_val << "]" << std::endl;
}

inline double norm(double3d_ptr a)
{
    PROFILE_FUNCTION();
    std::size_t n = a->num_elements();

    double ret = static_cast<double>(0);
    for ( std::size_t i = 0; i < n; ++i )
    {
        ret += a->data()[i]*a->data()[i];
    }

    return std::sqrt(ret);
}

inline double cross_correlation(double3d_ptr a, double3d_ptr b)
{
    PROFILE_FUNCTION();
    ASSERT_SAME_SIZE(a,b);
    double ret = static_cast<double>(0);

    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        ret += a->data()[i] * b->data()[i];
    }
    return ret;
}

inline void binarize(double3d_ptr a, double thresh = 0.5)
{
    PROFILE_FUNCTION();
    
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = (a->data()[i] > thresh) ?
            static_cast<double>(1):static_cast<double>(0);
    }
}

inline bool3d_ptr binary_mask(double3d_ptr a, double thresh = 0.5)
{
    PROFILE_FUNCTION();

    bool3d_ptr r = volume_pool.get_bool3d(size_of(a));    
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        r->data()[i] = (a->data()[i] > thresh) ? true : false;
    }
    return r;
}

template <typename T>
inline void random_initialization(T& generator, double3d_ptr a)
{
    PROFILE_FUNCTION();
    
    std::size_t n = a->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        a->data()[i] = generator();
    }
}

std::list<double3d_ptr> encode_multiclass( double3d_ptr label, std::size_t n_class )
{
    std::vector<double3d_ptr> vret;
    for ( std::size_t i = 0; i < n_class; ++i )
    {
        double3d_ptr lbl = volume_pools.get_double3d(label);
        volume_utils::zero_out(lbl);
        vret.push_back(lbl);
    }
    
    std::size_t n = label->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {            
        std::size_t idx = static_cast<std::size_t>(label->data()[i] + 0.5);
        STRONG_ASSERT( idx < n_class );
        vret[idx]->data()[i] = static_cast<double>(1);
    }   
    
    std::list<double3d_ptr> ret(vret.begin(),vret.end());
    return ret;
}

inline 
double3d_ptr mirror_boundary( double3d_ptr vol, vec3i RF )
{    
    std::size_t vx = vol->shape()[0];
    std::size_t vy = vol->shape()[1];
    std::size_t vz = vol->shape()[2];

    std::size_t mx = RF[0]/2;
    std::size_t my = RF[1]/2;
    std::size_t mz = RF[2]/2;

    std::size_t rx = vx + 2*mx;
    std::size_t ry = vy + 2*my;
    std::size_t rz = vz + 2*mz;

    double3d_ptr r = volume_pool.get_double3d(rx,ry,rz);
    volume_utils::zero_out(r);
    
    // copy original volume
    for ( std::size_t x = 0; x < vx; ++x )
        for ( std::size_t y = 0; y < vy; ++y )
            for ( std::size_t z = 0; z < vz; ++z )
                (*r)[x+mx][y+my][z+mz] = (*vol)[x][y][z];

    // x-direction
    for ( std::size_t x = 1; x <= mx; ++x )
        for ( std::size_t y = my; y < ry-my; ++y )
            for ( std::size_t z = mz; z < rz-mz; ++z )
            {
                (*r)[mx-x][y][z] = (*r)[mx+x][y][z];
                (*r)[mx+vx-1+x][y][z] = (*r)[mx+vx-1-x][y][z];
            }

    // y-direction
    for ( std::size_t y = 1; y <= my; ++y )
        for ( std::size_t x = 0; x < rx; ++x )
            for ( std::size_t z = mz; z < rz - mz; ++z )
            {
                (*r)[x][my-y][z] = (*r)[x][my+y][z];
                (*r)[x][my+vy-1+y][z] = (*r)[x][my+vy-1-y][z];
            }

    // z-direction
    for ( std::size_t z = 1; z <= mz; ++z )
        for ( std::size_t x = 0; x < rx; ++x )
            for ( std::size_t y = 0; y < ry; ++y )
            {
                (*r)[x][y][mz-z] = (*r)[x][y][mz+z];
                (*r)[x][y][mz+vz-1+z] = (*r)[x][y][mz+vz-1-z];
            }

    // handle boundary effect by even-sized receptive field
    vec3i margin = vec3i::zero;
    if ( RF[0] % 2 == 0 ) ++(margin[0]);
    if ( RF[1] % 2 == 0 ) ++(margin[1]);
    if ( RF[2] % 2 == 0 ) ++(margin[2]);
    if ( margin != vec3i::zero )
        r = volume_utils::crop(r,rx-margin[0],ry-margin[1],rz-margin[2]);

    return r;
}

inline double3d_ptr 
binomial_rebalance_mask( double3d_ptr lbl, double thresh = 0.5 )
{
    double3d_ptr pos = volume_pool.get_double3d(lbl);
    double3d_ptr neg = volume_pool.get_double3d(lbl);

    std::size_t n = lbl->num_elements();
    for ( std::size_t i = 0; i < n; ++i )
    {
        bool b = lbl->data()[i] > thresh;
        pos->data()[i] = b ? static_cast<double>(1):static_cast<double>(0);
        neg->data()[i] = b ? static_cast<double>(0):static_cast<double>(1);
    }

    double npos = volume_utils::sum_all(pos);
    double nneg = volume_utils::sum_all(neg);

    // return mask
    double3d_ptr ret = volume_pool.get_double3d(lbl);

    // avoid divide-by-zero case
    if ( npos < 1 || nneg < 1 )
    {
        volume_utils::fill_one(ret);
    }
    else
    {
        double wpos = static_cast<double>(1)/npos;
        double wneg = static_cast<double>(1)/nneg;
        double sum  = wpos + wneg;
        
        wpos /= sum; 
        wneg /= sum;
        
        volume_utils::zero_out(ret);
        volume_utils::mul_add_to(wpos,pos,ret);
        volume_utils::mul_add_to(wneg,neg,ret);
    }

    return ret;
}

inline std::list<double3d_ptr> 
binomial_rebalance_mask( std::list<double3d_ptr> lbls, double thresh = 0.5 )
{
    std::list<double3d_ptr> ret;

    FOR_EACH( it, lbls )
    {
        ret.push_back(volume_utils::binomial_rebalance_mask(*it,thresh));
    }

    return ret;
}

inline double3d_ptr
multinomial_rebalance_mask( std::list<double3d_ptr> lbls )
{
    bool divide_by_zero = false;

    // rebalancing weights
    std::vector<double> weights;
    double sum = static_cast<double>(0);
    FOR_EACH( it, lbls )
    {
        double n = volume_utils::sum_all(*it);
        if ( n < 1 ) divide_by_zero = true;
        double w = static_cast<double>(1)/n;
        weights.push_back(w);
        sum += w;
    }

    // normalize rebalancing weights and construct weight mask
    double3d_ptr ret = volume_pool.get_double3d(lbls.front());

    if ( divide_by_zero )
    {
        volume_utils::fill_one(ret);
    }
    else
    {
        volume_utils::zero_out(ret);
        
        std::size_t idx = 0;
        FOR_EACH( it, lbls )
        {
            double w = weights[idx++]/sum;
            volume_utils::mul_add_to(w,*it,ret);
        }
    }

    return ret;
}

}; // abstract class volume_utils

}} // namespace zi::znn::volume_utils

#endif // ZNN_VOLUME_UTILS_HPP_INCLUDED
