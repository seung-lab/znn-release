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

#include "../../types.hpp"
#include "../../assert.hpp"

namespace znn { namespace v4 {

#define __ZNN_ALIGN 0xF // 16 byte alignment

inline void* znn_malloc(size_t s)
{
    void* r = malloc(s);
    if ( !r ) throw std::bad_alloc();
    return r;
}

inline void znn_free(void* ptr)
{
    free(ptr);
}

template <typename T> struct cube: boost::multi_array_ref<T,3>
{
private:
    using base_type =  boost::multi_array_ref<T,3>;

public:
    explicit cube(const vec3i& s, T* data)
        : boost::multi_array_ref<T,3>(data,extents[s[0]][s[1]][s[2]])
    {
    }

    ~cube()
    {
        DIE("should never be called");
    }

    cube& operator=(const cube& x)
    {
        base_type::operator=(static_cast<base_type>(x));
        return *this;
    }

    template< class Array >
    cube& operator=(const Array& x)
    {
        base_type::operator=(x);
        return *this;
    }

};


template <typename T> struct qube: boost::multi_array_ref<T,4>
{
private:
    using base_type =  boost::multi_array_ref<T,4>;

public:
    explicit qube(const vec4i& s, T* data)
        : boost::multi_array_ref<T,4>(data,extents[s[0]][s[1]][s[2]][s[3]])
    {
    }

    ~qube()
    {
        DIE("should never be called");
    }

    qube& operator=(const qube& x)
    {
        base_type::operator=(static_cast<base_type>(x));
        return *this;
    }

    template< class Array >
    qube& operator=(const Array& x)
    {
        base_type::operator=(x);
        return *this;
    }
};

template<typename T>
struct __znn_aligned_size
{
    static const size_t value = ((sizeof(T)-1) | __ZNN_ALIGN) + 1;
    static_assert((value&__ZNN_ALIGN)==0, "bad value");
};

inline void* znn_aligned_malloc(size_t s)
{
    return znn_malloc(((s-1)|__ZNN_ALIGN)+1);
}

template<class T>
inline T* __offset_cast(void* mem, size_t off)
{
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mem)+off);
}

template<typename T>
std::shared_ptr<cube<T>> get_cube(const vec3i& s)
{
    void*    mem  = znn_aligned_malloc(__znn_aligned_size<cube<T>>::value
                                       + s[0]*s[1]*s[2]*sizeof(T) );

    ZI_ASSERT((reinterpret_cast<size_t>(mem)&__ZNN_ALIGN)==0);

    T*       data = __offset_cast<T>(mem, __znn_aligned_size<cube<T>>::value);
    cube<T>* c    = new (mem) cube<T>(s,data);

    ZI_ASSERT(c==mem);

    return std::shared_ptr<cube<T>>(c,znn_free);
}

template<typename T>
std::shared_ptr<qube<T>> get_qube(const vec4i& s)
{
    void*    mem  = znn_aligned_malloc(__znn_aligned_size<qube<T>>::value
                                       + s[0]*s[1]*s[2]*s[3]*sizeof(T));

    ZI_ASSERT((reinterpret_cast<size_t>(mem)&__ZNN_ALIGN)==0);

    T*       data = __offset_cast<T>(mem,__znn_aligned_size<qube<T>>::value);
    qube<T>* c    = new (mem) qube<T>(s,data);

    return std::shared_ptr<qube<T>>(c,znn_free);
}

}} // namespace znn::v4
