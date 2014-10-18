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

#ifndef ZNN_UTILS_HPP_INCLUDED
#define ZNN_UTILS_HPP_INCLUDED

#include "types.hpp"
#include <zi/zpp/stringify.hpp>
#include <boost/lexical_cast.hpp>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace zi {
namespace znn {

template <typename T>
inline vec3i size_of( const T& a )
{
    return vec3i(a->shape()[0],a->shape()[1],a->shape()[2]);
};

void export_size_info( const vec3i& sz, const std::string& fname )
{
    std::string ssz = fname + ".size";
    std::ofstream fsz(ssz.c_str(), (std::ios::out | std::ios::binary) );
    
    uint32_t u;
    u = static_cast<uint32_t>(sz[0]);
    fsz.write( reinterpret_cast<char*>(&u), sizeof(uint32_t) );
    u = static_cast<uint32_t>(sz[1]);
    fsz.write( reinterpret_cast<char*>(&u), sizeof(uint32_t) );
    u = static_cast<uint32_t>(sz[2]);
    fsz.write( reinterpret_cast<char*>(&u), sizeof(uint32_t) );
}

vec3i import_size_info( const std::string& fname )
{
    vec3i ret = vec3i::zero;
    std::string ssz = fname + ".size";
    FILE* fp = fopen(ssz.c_str(), "r");
    
    // size of the whole volume
    if ( fp )
    {
        uint32_t sz[3];
        static_cast<void>(fread(sz, sizeof(uint32_t), 3, fp));
        ret = vec3i(sz[0],sz[1],sz[2]);
    }

    return ret;
}

std::size_t count_digit( std::size_t n )
{
    std::size_t cnt = 0;
    while ( (n /= 10) > 0 ) cnt++;
    return cnt;
}

template<class T>
bool almost_equal( T x, T y, int ulp )
{
    return std::abs(x-y) <=   std::numeric_limits<T>::epsilon()
                            * std::max(std::abs(x), std::abs(y))
                            * ulp;
}

std::string vec3i_to_string( const vec3i& v )
{
    std::string ret = boost::lexical_cast<std::string>(v[0]) + "," +
                      boost::lexical_cast<std::string>(v[1]) + "," +
                      boost::lexical_cast<std::string>(v[2]);
    return ret;
}

std::string strip_brackets( const std::string& s )
{
    std::size_t len = s.size();

    if ( len < 2 )
    {
        return s;
    }

    if ( (s[0] == '[') && (s[len-1] == ']') )
    {
        std::string ret(s.begin()+1, s.end()-1);
        return ret;
    }

    return s;
}

}} // namespace zi::znn

#define STRONG_ASSERT(condition)                                        \
    if (!(condition))                                                   \
    {                                                                   \
        std::cout << "Assertion " << ZiPP_STRINGIFY(condition) << " failed " \
                  << "file: " << __FILE__ << " line: " << __LINE__ << std::endl; \
        abort();                                                        \
    }                                                                   \
    static_cast<void>(0)

#define MILD_ASSERT(condition)                                          \
    if (!(condition))                                                   \
    {                                                                   \
        std::cout << "Assertion " << ZiPP_STRINGIFY(condition) << " failed " \
                  << "file: " << __FILE__ << " line: " << __LINE__ << std::endl; \
    }                                                                   \
    static_cast<void>(0)

#define ASSERT_SAME_SIZE(a,b)                                   \
    STRONG_ASSERT(::zi::znn::volume_utils::volume_size(a)==     \
                  ::zi::znn::volume_utils::volume_size(b))

template <class T>
struct sort_functor
{
    bool operator()(T& a, T& b)
    {
        return (*a) < (*b);
    }
};

#endif // ZNN_UTILS_HPP_INCLUDED
