#pragma once

#include <zi/vl/vl.hpp>
#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>


namespace znn { namespace v4 {

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


typedef ovec<int64_t,2> ovec2i;
typedef ovec<int64_t,3> ovec3i;
typedef ovec<int64_t,4> ovec4i;
typedef ovec<real,2>  ovec2d;
typedef ovec<real,3>  ovec3d;
typedef ovec<real,4>  ovec4d;

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


}} // namespace znn::v4
