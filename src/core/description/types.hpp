//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_DESCRIPTION_TYPES_HPP_INCLUDED
#define ZNN_CORE_DESCRIPTION_TYPES_HPP_INCLUDED

#include <zi/vl/vl.hpp>
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

namespace zi {
namespace znn {

template< typename T, std::size_t N >
struct ovec: zi::vl::vec<T,N> {};

template< class T, std::size_t N, class CharT, class Traits >
::std::basic_istream< CharT, Traits >&
operator>>( ::std::basic_istream< CharT, Traits >& is,
            ovec< T, N >& v )
{
    std::string s;
    is >> s;

    std::vector<std::string> parts;
    boost::split(parts, s, boost::is_any_of(","));

    for ( std::size_t i = 0; i < std::min(N,parts.size()); ++i )
    {
        v[i] = boost::lexical_cast<T>(parts[i]);
    }

    return is;
}

template< class T, std::size_t N, class CharT, class Traits >
::std::basic_ostream< CharT, Traits >&
operator<<( ::std::basic_ostream< CharT, Traits >& os,
            const ovec< T, N >& v )
{
    os << v[ 0 ];
    for ( std::size_t i = 1; i < N; ++i )
    {
        os << "," << v[ i ];
    }
    return os;
}


typedef ovec<std::size_t,3> ovec3i;

template< typename T, typename Allocator = std::allocator<T> >
struct ovector: std::vector<T,Allocator> {};


template< class T, class Allocator, class CharT, class Traits >
::std::basic_istream< CharT, Traits >&
operator>>( ::std::basic_istream< CharT, Traits >& is,
            ovector<T,Allocator>& v )
{
    std::string s;
    is >> s;
    v.clear();

    std::vector<std::string> parts;
    boost::split(parts, s, boost::is_any_of(","));

    for ( std::size_t i = 0; i < parts.size(); ++i )
    {
        v.push_back(boost::lexical_cast<T>(parts[i]));
    }

    return is;
}

template< class T, class Allocator, class CharT, class Traits >
::std::basic_ostream< CharT, Traits >&
operator<<( ::std::basic_ostream< CharT, Traits >& os,
            const ovector<T,Allocator>& v )
{
    if ( v.size() )
    {
        os << v[ 0 ];
    }

    for ( std::size_t i = 1; i < v.size(); ++i )
    {
        os << "," << v[ i ];
    }
    return os;
}


}} // namespace zi::znn


#endif // ZNN_CORE_DESCRIPTION_TYPES_HPP_INCLUDED
