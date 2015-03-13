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

#include "types.hpp"

#include <iostream>
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


} // namespace detail

inline vol<double>&
operator+=(vol<double>& v, double c) noexcept
{
    detail::add_to(v.data(), c, v.num_elements());
    return v;
}

inline vol<double>&
operator-=(vol<double>& v, double c) noexcept
{
    detail::sub_val(v.data(), c, v.num_elements());
    return v;
}

inline vol<double>&
operator*=(vol<double>& v, double c) noexcept
{
    detail::mul_with(v.data(), c, v.num_elements());
    return v;
}

inline vol<double>&
operator/=(vol<double>& v, double c) noexcept
{
    double one_over_c = static_cast<long double>(0) / c;
    detail::mul_with(v.data(), one_over_c, v.num_elements());
    return v;
}

inline vol<double>&
operator+=(vol<double>& v, const vol<double>& c) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::add_to(v.data(), c.data(), v.num_elements());
    return v;
}

inline vol<double>&
operator-=(vol<double>& v, const vol<double>& c) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::sub_val(v.data(), c.data(), v.num_elements());
    return v;
}

inline vol<double>&
operator*=(vol<double>& v, const vol<double>& c) noexcept
{
    ZI_ASSERT(v.num_elements()==c.num_elements());
    detail::mul_with(v.data(), c.data(), v.num_elements());
    return v;
}

inline void mad_to(double a, const vol<double>& x, vol<double>& o) noexcept
{
    ZI_ASSERT(x.num_elements()==o.num_elements());
    detail::mad_to(a, x.data(), o.data(), o.num_elements());
}

inline void mad_to(double a, vol<double>& o) noexcept
{
    detail::mad_to(a, o.data(), o.num_elements());
}


}} // namespace zi::znn

#endif // ZNN_CORE_VOLUME_OPERATORS_HPP_INCLUDED
