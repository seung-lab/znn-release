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

#include <cstdint>
#include <cstddef>
#include <complex>
#include <mutex>
#include <memory>
#include <functional>
#include <zi/vl/vl.hpp>
#include <boost/multi_array.hpp>
#include <lockfree_allocator.hpp>

#include <map>
#include <list>
#include <vector>


#if ( __cplusplus <= 201103L )

namespace std {

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}

#ifdef ZNN_NO_THREAD_LOCAL
#  define ZNN_THREAD_LOCAL
#else
#  define ZNN_THREAD_LOCAL thread_local
#endif

#endif

namespace znn { namespace v4 {

#ifdef ZNN_USE_FLOATS
typedef float                  real;
#else
typedef double                 real;
#endif

typedef std::complex<real>   cplx;
typedef std::complex<real>   complex;

typedef zi::vl::vec<int64_t,3> vec3i;
typedef zi::vl::vec<int64_t,4> vec4i;


typedef std::size_t size_t;

typedef std::lock_guard<std::mutex> guard;

typedef int64_t long_t;

typedef boost::multi_array_types::index_range range;

namespace { decltype(boost::indices) indices; }
namespace { decltype(boost::extents) extents; }

template< class > struct vec_hash;

template< class T, size_t N>
struct vec_hash<zi::vl::vec<T,N>>
{
    size_t operator()(zi::vl::vec<T,N> const & s) const
    {
        std::hash<T> hasher;
        size_t seed = hasher(s[0]);

        for ( size_t i = 1; i < N; ++i )
        {
            seed ^= s[i] + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }
};


template<class K, class V>
using map = std::map<K,V,std::less<K>,allocator<std::pair<const K,V>>>;

template<class T>
using vector = std::vector<T,allocator<T>>;

template<class T>
using list = std::list<T,allocator<T>>;


template<class T>
struct unique_ptr_deleter
{
    unique_ptr_deleter() noexcept {}

    void operator()(T* p) const
    {
        allocator<T> alloc;
        alloc.destroy(p);
        alloc.deallocate(p);
    }
};

template<typename T>
using unique_ptr = std::unique_ptr<T,unique_ptr_deleter<T>>;

template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args)
{
    allocator<T> alloc;
    T* p = alloc.allocate(1);
    alloc.construct(p, std::forward<Args>(args)...);
    return unique_ptr<T>(p);
}


}} // namespace znn::v4
