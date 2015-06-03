#pragma once

#include "cube.hpp"
#include "../types.hpp"
#include "../meta.hpp"

#include <iostream>
#include <algorithm>
#include <type_traits>

namespace znn { namespace v4 {

template< class T, class CharT, class Traits >
std::basic_ostream< CharT, Traits >&
operator<<( ::std::basic_ostream< CharT, Traits >& os,
            cube<T> const & a )
{
    std::size_t rx = a.shape()[0];
    std::size_t ry = a.shape()[1];
    std::size_t rz = a.shape()[2];

    for ( std::size_t z = 0; z < rz; ++z )
    {
        if ( z > 0 )
        {
            os << "\n\n";
        }
        for ( std::size_t x = 0; x < rx; ++x )
        {
            if ( x > 0 )
            {
                os << "\n";
            }
            for ( std::size_t y = 0; y < ry; ++y )
            {
                if ( y > 0 )
                {
                    os << ' ';
                }
                os << a[x][y][z];
            }
        }
    }
    return os;
}


namespace detail {

template<typename T>
inline void add_two(T const * a, T const * b, T * r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] + b[i];
}

template<typename T>
inline void sub_two(T const * a, T const * b, T * r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] - b[i];
}

template<typename T>
inline void mul_two(T const * a, T const * b, T * r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] * b[i];
}

template<typename T>
inline void div_two(T const * a, T const * b, T * r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] / b[i];
}

template<typename T>
inline void add_to(T * a, T const& v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] += v;
}

template<typename T>
inline void sub_val(T * a, T const& v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] -= v;
}

template<typename T>
inline void mul_with(T * a, T const& v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] *= v;
}

template<typename T>
inline void div_with(T * a, T const& v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] /= v;
}

template<typename T>
inline void add_to(T * a, T const * v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] += v[i];
}

template<typename T>
inline void mad_to(real a, T const * x, T * o, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        o[i] += a * x[i];
}

template<typename T>
inline void mad_to(T const * a, T const * b, T * r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] += a[i] * b[i];
}

template<typename T>
inline void mad_to(real a, T * o, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        o[i] += a * o[i];
}

template<typename T>
inline void sub_val(T * a, T const * v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] -= v[i];
}

template<typename T>
inline void mul_with(T * a, T const * v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] *= v[i];
}

template<typename T>
inline T sum(T const * a, std::size_t s) noexcept
{
    T r = T();
    for ( std::size_t i = 0; i < s; ++i )
        r += a[i];
    return r;
}


} // namespace detail

template<typename T>
inline cube<T> &
operator+=( cube<T> & v, identity_t<T> c ) noexcept
{
    detail::add_to(v.data(), c, v.num_elements());
    return v;
}

template<typename T>
inline cube<T> &
operator-=( cube<T> & v, identity_t<T> c ) noexcept
{
    detail::sub_val(v.data(), c, v.num_elements());
    return v;
}

template<typename T>
inline cube<T> &
operator*=( cube<T> & v, identity_t<T> c ) noexcept
{
    detail::mul_with(v.data(), c, v.num_elements());
    return v;
}

template<typename T>
inline cube<T> &
operator/=( cube<T> & v, identity_t<T> c ) noexcept
{
    real one_over_c = static_cast<long double>(1) / c;
    detail::mul_with(v.data(), one_over_c, v.num_elements());
    //detail::div_with(v.data(), c, v.num_elements());
    return v;
}

template<typename T>
inline cube<T> &
operator+=( cube<T> & v, cube<T> const & c ) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::add_to(v.data(), c.data(), v.num_elements());
    return v;
}

template<typename T>
inline cube<T> &
operator-=( cube<T> & v, cube<T> const & c ) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::sub_val(v.data(), c.data(), v.num_elements());
    return v;
}

template<typename T>
inline cube<T> &
operator*=( cube<T> & v, cube<T> const & c ) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::mul_with(v.data(), c.data(), v.num_elements());
    return v;
}


template<typename T>
inline cube_p<T>
operator+( cube<T> const & a, cube<T> const & b)
{
    ZI_ASSERT(size(a)==size(b));
    cube_p<T> r = get_cube<T>(size(a));
    detail::add_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline cube_p<T>
operator-( cube<T> const & a, cube<T> const & b)
{
    ZI_ASSERT(size(a)==size(b));
    cube_p<T> r = get_cube<T>(size(a));
    detail::sub_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline cube_p<T>
operator*( cube<T> const & a, cube<T> const & b)
{
    ZI_ASSERT(size(a)==size(b));
    cube_p<T> r = get_cube<T>(size(a));
    detail::mul_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline cube_p<T>
operator/( cube<T> const & a, cube<T> const & b)
{
    ZI_ASSERT(size(a)==size(b));
    cube_p<T> r = get_cube<T>(size(a));
    detail::div_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline void mad_to(identity_t<T> a,
                   cube<T> const & x, cube<T> & o) noexcept
{
    ZI_ASSERT(x.num_elements()==o.num_elements());
    detail::mad_to(a, x.data(), o.data(), o.num_elements());
}

template<typename T>
inline void mad_to(cube<T> const & a, cube<T> const & b, cube<T> & o) noexcept
{
    ZI_ASSERT(a.num_elements()==b.num_elements());
    ZI_ASSERT(b.num_elements()==o.num_elements());
    detail::mad_to(a.data(), b.data(), o.data(), o.num_elements());
}


template<typename T>
inline void mad_to(identity_t<T> a, cube<T> & o) noexcept
{
    detail::mad_to(a, o.data(), o.num_elements());
}

template< typename T >
inline void fill( cube<T> & v, identity_t<T> c) noexcept
{
    std::fill_n(v.data(), v.num_elements(), c);
}

inline void flip(cube<real>& v) noexcept
{
    real* data = v.data();
    std::reverse(data, data + v.num_elements());
}

template<typename T>
inline T max(cube<T> const & v) noexcept
{
    return *std::max_element(v.data(), v.data() + v.num_elements());
}

template<typename T>
inline T min(cube<T> const & v) noexcept
{
    return *std::min_element(v.data(), v.data() + v.num_elements());
}

template<typename T>
inline T sum(cube<T> const & v) noexcept
{
    return detail::sum(v.data(), v.num_elements());
}


template<typename T>
inline cube_p<T> sparse_explode( cube<T> const & v,
                                 vec3i const & sparse,
                                 vec3i const & s )
{
    vec3i vs = size(v);
    cube_p<T> r = get_cube<T>(s);
    fill(*r,0);

    (*r)[indices
         [range(0,sparse[0]*vs[0],sparse[0])]
         [range(0,sparse[1]*vs[1],sparse[1])]
         [range(0,sparse[2]*vs[2],sparse[2])]] = v;

    return r;
}

template<typename T>
inline cube_p<T> sparse_explode_slow( cube<T> const & v,
                                      vec3i const & sparse,
                                      vec3i const & s )
{
    vec3i vs = size(v);
    cube_p<T> r = get_cube<T>(s);
    fill(*r,0);

    cube<T> & rr = *r;

    for ( long_t xv = 0, rx = 0; xv < vs[0]; ++xv, rx += sparse[0] )
        for ( long_t yv = 0, ry = 0; yv < vs[1]; ++yv, ry += sparse[1] )
            for ( long_t zv = 0, rz = 0; zv < vs[2]; ++zv, rz += sparse[2] )
                rr[rx][ry][rz] = v[xv][yv][zv];

    return r;
}


template<typename T>
inline cube_p<T> sparse_implode_slow( cube<T> const & r,
                                      vec3i const & sparse,
                                      vec3i const & vs )
{
    cube_p<T> vp = get_cube<T>(vs);
    cube<T> & v = *vp;

    for ( long_t xv = 0, rx = 0; xv < vs[0]; ++xv, rx += sparse[0] )
        for ( long_t yv = 0, ry = 0; yv < vs[1]; ++yv, ry += sparse[1] )
            for ( long_t zv = 0, rz = 0; zv < vs[2]; ++zv, rz += sparse[2] )
                v[xv][yv][zv] = r[rx][ry][rz];

    return vp;
}

template<typename T>
inline cube_p<T> sparse_implode( cube<T> const & r,
                                 vec3i const & sparse,
                                 vec3i const & vs )
{
    auto vp = get_cube<T>(vs);
    (*vp) = r[indices
              [range(0,sparse[0]*vs[0],sparse[0])]
              [range(0,sparse[1]*vs[1],sparse[1])]
              [range(0,sparse[2]*vs[2],sparse[2])]];
    return vp;
}


inline cube_p<real> pad_zeros( const cube<real>& v, vec3i const & s )
{
    cube_p<real> r = get_cube<real>(s);

    std::size_t ox = v.shape()[0];
    std::size_t oy = v.shape()[1];
    std::size_t oz = v.shape()[2];

    if ( size(v) != s ) fill(*r, 0);

    (*r)[indices[range(0,ox)][range(0,oy)][range(0,oz)]] = v;

    return r;
}

template<typename T>
inline cube_p<T> crop( cube<T> const & c, vec3i const & l, vec3i const & s )
{
    auto ret = get_cube<T>(s);
    *ret = c[indices
             [range(l[0],l[0]+s[0])]
             [range(l[1],l[1]+s[1])]
             [range(l[2],l[2]+s[2])]];
    return ret;
}

template<typename T>
inline cube_p<T> crop( cube<T> const & c, vec3i const & s )
{
    auto ret = get_cube<T>(s);
    *ret = c[indices[range(0,s[0])][range(0,s[1])][range(0,s[2])]];
    return ret;
}

template<typename T>
inline cube_p<T> crop_left( cube<T> const & c, vec3i const & s )
{
    return crop(c,s);
}


template<typename T>
inline cube_p<T> crop_right( cube<T> const & c, vec3i const & s )
{
    vec3i off = size(c) - s;
    auto ret = get_cube<T>(s);
    *ret = c[indices
             [range(off[0],s[0]+off[0])]
             [range(off[1],s[1]+off[1])]
             [range(off[2],s[2]+off[2])]];
    return ret;
}

template<typename T>
inline void flatten( cube<T> & c, vec3i const & s )
{
    vec3i sz = size(c);
    ZI_ASSERT((sz%s)==vec3i::zero);

    T n_elem = s[0] * s[1] * s[2];

    for ( long_t z = 0; z < sz[2]; z += s[2] )
        for ( long_t y = 0; y < sz[1]; y += s[1] )
            for ( long_t x = 0; x < sz[0]; x += s[0] )
            {
                T r = 0;
                for ( long_t zi = 0; zi < s[2]; ++zi )
                    for ( long_t yi = 0; yi < s[1]; ++yi )
                        for ( long_t xi = 0; xi < s[0]; ++xi )
                            r += c[x+xi][y+yi][z+zi];

                r /= n_elem;

                for ( long_t zi = 0; zi < s[2]; ++zi )
                    for ( long_t yi = 0; yi < s[1]; ++yi )
                        for ( long_t xi = 0; xi < s[0]; ++xi )
                            c[x+xi][y+yi][z+zi] = r;
            }
}

template<typename T>
inline cube_p<T> mirror_boundary( cube<T> const & c,
                                  vec3i const & rf )
{
    STRONG_ASSERT(rf[0] % 2);
    STRONG_ASSERT(rf[1] % 2);
    STRONG_ASSERT(rf[2] % 2);

    long_t vx = c.shape()[0];
    long_t vy = c.shape()[1];
    long_t vz = c.shape()[2];

    long_t mx = rf[0]/2;
    long_t my = rf[1]/2;
    long_t mz = rf[2]/2;

    long_t rx = vx + 2*mx;
    long_t ry = vy + 2*my;
    long_t rz = vz + 2*mz;

    auto rp = get_cube<T>(vec3i(rx,ry,rz));
    cube<T>& r = *rp;

    r = 0;

    // copy original volume
    for ( long_t x = 0; x < vx; ++x )
        for ( long_t y = 0; y < vy; ++y )
            for ( long_t z = 0; z < vz; ++z )
                r[x+mx][y+my][z+mz] = c[x][y][z];

    // x-direction
    for ( long_t x = 1; x <= mx; ++x )
        for ( long_t y = my; y < ry-my; ++y )
            for ( long_t z = mz; z < rz-mz; ++z )
            {
                r[mx-x][y][z] = r[mx+x][y][z];
                r[mx+vx-1+x][y][z] = r[mx+vx-1-x][y][z];
            }

    // y-direction
    for ( long_t y = 1; y <= my; ++y )
        for ( long_t x = 0; x < rx; ++x )
            for ( long_t z = mz; z < rz - mz; ++z )
            {
                r[x][my-y][z] = r[x][my+y][z];
                r[x][my+vy-1+y][z] = r[x][my+vy-1-y][z];
            }

    // z-direction
    for ( long_t z = 1; z <= mz; ++z )
        for ( long_t x = 0; x < rx; ++x )
            for ( long_t y = 0; y < ry; ++y )
            {
                r[x][y][mz-z] = r[x][y][mz+z];
                r[x][y][mz+vz-1+z] = r[x][y][mz+vz-1-z];
            }

    return rp;
}

}} // namespace znn::v4
