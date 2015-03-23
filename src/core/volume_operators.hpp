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

#ifndef ZNN_CORE_VOLUME_OPERATORS_HPP_INCLUDED
#define ZNN_CORE_VOLUME_OPERATORS_HPP_INCLUDED

#include "volume_pool.hpp"
#include "types.hpp"

#include <iostream>
#include <algorithm>
#include <type_traits>

namespace zi {
namespace znn {

template< class T, class CharT, class Traits >
std::basic_ostream< CharT, Traits >&
operator<<( ::std::basic_ostream< CharT, Traits >& os,
            const vol<T>& a )
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
    return os << "\n\n";
}

namespace detail {


template<typename T>
inline void add_two(const T* a, const T* b, T* r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] + b[i];
}

template<typename T>
inline void sub_two(const T* a, const T* b, T* r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] - b[i];
}

template<typename T>
inline void mul_two(const T* a, const T* b, T* r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] * b[i];
}

template<typename T>
inline void div_two(const T* a, const T* b, T* r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] = a[i] / b[i];
}

template<typename T>
inline void add_to(T* a, const T& v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] += v;
}

template<typename T>
inline void sub_val(T* a, const T& v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] -= v;
}

template<typename T>
inline void mul_with(T* a, const T& v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] *= v;
}

template<typename T>
inline void add_to(T* a, const T* v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] += v[i];
}

template<typename T>
inline void mad_to(double a, const T* x, T* o, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        o[i] += a * x[i];
}

template<typename T>
inline void mad_to(const T* a, const T* b, T* r, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        r[i] += a[i] * b[i];
}

template<typename T>
inline void mad_to(double a, T* o, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        o[i] += a * o[i];
}

template<typename T>
inline void sub_val(T* a, const T* v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] -= v[i];
}

template<typename T>
inline void mul_with(T* a, const T* v, std::size_t s) noexcept
{
    for ( std::size_t i = 0; i < s; ++i )
        a[i] *= v[i];
}

template<typename T>
inline T sum(const T* a, std::size_t s) noexcept
{
    T r = T();
    for ( std::size_t i = 0; i < s; ++i )
        r += a[i];
    return r;
}


} // namespace detail

template<typename T>
inline vol<T>&
operator+=(vol<T>& v, typename identity<T>::type c) noexcept
{
    detail::add_to(v.data(), c, v.num_elements());
    return v;
}

template<typename T>
inline vol<T>&
operator-=(vol<T>& v, typename identity<T>::type c) noexcept
{
    detail::sub_val(v.data(), c, v.num_elements());
    return v;
}

template<typename T>
inline vol<T>&
operator*=(vol<T>& v, typename identity<T>::type c) noexcept
{
    detail::mul_with(v.data(), c, v.num_elements());
    return v;
}

template<typename T>
inline vol<T>&
operator/=(vol<T>& v, typename identity<T>::type c) noexcept
{
    double one_over_c = static_cast<long double>(1) / c;
    detail::mul_with(v.data(), one_over_c, v.num_elements());
    return v;
}

template<typename T>
inline vol<T>&
operator+=(vol<T>& v, const vol<T>& c) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::add_to(v.data(), c.data(), v.num_elements());
    return v;
}

template<typename T>
inline vol<T>&
operator-=(vol<T>& v, const vol<T>& c) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::sub_val(v.data(), c.data(), v.num_elements());
    return v;
}

template<typename T>
inline vol<T>&
operator*=(vol<T>& v, const vol<T>& c) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::mul_with(v.data(), c.data(), v.num_elements());
    return v;
}


template<typename T>
inline vol_p<T>
operator+(const vol<T>& a, const vol<T>& b)
{
    ZI_ASSERT(size(a)==size(b));
    vol_p<T> r = get_volume<T>(size(a));
    detail::add_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline vol_p<T>
operator-(const vol<T>& a, const vol<T>& b)
{
    ZI_ASSERT(size(a)==size(b));
    vol_p<T> r = get_volume<T>(size(a));
    detail::sub_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline vol_p<T>
operator*(const vol<T>& a, const vol<T>& b)
{
    ZI_ASSERT(size(a)==size(b));
    vol_p<T> r = get_volume<T>(size(a));
    detail::mul_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline vol_p<T>
operator/(const vol<T>& a, const vol<T>& b)
{
    ZI_ASSERT(size(a)==size(b));
    vol_p<T> r = get_volume<T>(size(a));
    detail::div_two(a.data(), b.data(), r->data(), a.num_elements());
    return r;
}

template<typename T>
inline void mad_to(typename identity<T>::type a,
                   const vol<T>& x, vol<T>& o) noexcept
{
    ZI_ASSERT(x.num_elements()==o.num_elements());
    detail::mad_to(a, x.data(), o.data(), o.num_elements());
}

template<typename T>
inline void mad_to(const vol<T>& a, const vol<T>& b, vol<T>& o) noexcept
{
    ZI_ASSERT(a.num_elements()==b.num_elements());
    ZI_ASSERT(b.num_elements()==o.num_elements());
    detail::mad_to(a.data(), b.data(), o.data(), o.num_elements());
}


template<typename T>
inline void mad_to(typename identity<T>::type a, vol<T>& o) noexcept
{
    detail::mad_to(a, o.data(), o.num_elements());
}

template< typename T,
          class = typename
          std::enable_if<std::is_convertible<T,double>::value>::type >
inline void fill( vol<T>& v, const typename identity<T>::type & c) noexcept
{
    std::fill_n(v.data(), v.num_elements(), c);
}


inline void flip_vol(vol<double>& v) noexcept
{
    double* data = v.data();
    std::reverse(data, data + v.num_elements());
}

template<typename T>
inline T max(const vol<T>& v) noexcept
{
    return *std::max_element(v.data(), v.data() + v.num_elements());
}

template<typename T>
inline T min(const vol<T>& v) noexcept
{
    return *std::min_element(v.data(), v.data() + v.num_elements());
}

template<typename T>
inline T sum(const vol<T>& v) noexcept
{
    return detail::sum(v.data(), v.num_elements());
}


template<typename T>
inline vol_p<T> sparse_explode(const vol<T>& v, const vec3i& sparse,
                               const vec3i& s )
{
    vec3i vs = size(v);
    vol_p<T> r = get_volume<T>(s);
    fill(*r,0);

    vol<T>& rr = *r;

    for ( size_t xv = 0, rx = 0; xv < vs[0]; ++xv, rx += sparse[0] )
        for ( size_t yv = 0, ry = 0; yv < vs[1]; ++yv, ry += sparse[1] )
            for ( size_t zv = 0, rz = 0; zv < vs[2]; ++zv, rz += sparse[2] )
                rr[rx][ry][rz] = v[xv][yv][zv];

    return r;
}


template<typename T>
inline vol_p<T> sparse_implode(const vol<T>& r, const vec3i& sparse,
                               const vec3i& vs )
{
    vol_p<T> vp = get_volume<T>(vs);
    vol<T>& v = *vp;

    for ( size_t xv = 0, rx = 0; xv < vs[0]; ++xv, rx += sparse[0] )
        for ( size_t yv = 0, ry = 0; yv < vs[1]; ++yv, ry += sparse[1] )
            for ( size_t zv = 0, rz = 0; zv < vs[2]; ++zv, rz += sparse[2] )
                v[xv][yv][zv] = r[rx][ry][rz];

    return vp;
}


inline vol_p<double> pad_zeros( const cvol<double>& v, const vec3i& s )
{
    vol_p<double> r = get_volume<double>(s);

    std::size_t ox = v.shape()[0];
    std::size_t oy = v.shape()[1];
    std::size_t oz = v.shape()[2];

    if ( size(v) != s ) fill(*r, 0);

    (*r)[boost::indices[range(0,ox)][range(0,oy)][range(0,oz)]] = v;

    return r;
}

}} // namespace zi::znn

#endif // ZNN_CORE_VOLUME_OPERATORS_HPP_INCLUDED
