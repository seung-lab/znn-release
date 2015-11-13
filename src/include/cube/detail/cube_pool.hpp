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

#include <zi/utility/singleton.hpp>
#include <list>

#include "../../types.hpp"

#ifdef ZNN_XEON_PHI
#  include <mkl.h>
#endif

namespace znn { namespace v4 {

#ifdef ZNN_XEON_PHI
#  define __ZNN_ALIGN 0x3F // 64 byte alignment
#else
#  define __ZNN_ALIGN 0xF // 16 byte alignment
#endif

#ifdef ZNN_XEON_PHI

inline void* znn_malloc(size_t s)
{
    void* r = mkl_malloc(s,64);
    if ( !r ) throw std::bad_alloc();
    return r;
}

inline void znn_free(void* ptr)
{
    mkl_free(ptr);
}

#else

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

#endif

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
        znn_free(this);
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
    using base_type =  boost::multi_array_ref<T,3>;


public:
    explicit qube(const vec4i& s, T* data)
        : boost::multi_array_ref<T,4>(data,extents[s[0]][s[1]][s[2]][s[3]])
    {
    }

    ~qube()
    {
        znn_free(this);
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
cube<T>* malloc_cube(const vec3i& s)
{
    void*    mem  = znn_aligned_malloc(__znn_aligned_size<cube<T>>::value
                                       + s[0]*s[1]*s[2]*sizeof(T) );

    ZI_ASSERT((reinterpret_cast<size_t>(mem)&__ZNN_ALIGN)==0);

    T*       data = __offset_cast<T>(mem, __znn_aligned_size<cube<T>>::value);
    cube<T>* c    = new (mem) cube<T>(s,data);

    ZI_ASSERT(c==mem);

    return c;
}


template<typename T>
cube<T>* malloc_qube(const vec4i& s)
{
    void*    mem  = znn_aligned_malloc(__znn_aligned_size<qube<T>>::value
                                       + s[0]*s[1]*s[2]*s[3]*sizeof(T) );

    ZI_ASSERT((reinterpret_cast<size_t>(mem)&__ZNN_ALIGN)==0);

    T*       data = __offset_cast<T>(mem, __znn_aligned_size<qube<T>>::value);
    qube<T>* c    = new (mem) qube<T>(s,data);

    ZI_ASSERT(c==mem);

    return c;
}


template<typename T>
class single_size_cube_pool
{
private:
    vec3i                      size_;
    std::list<cube<T>*>        list_;
    std::mutex                 m_   ;

public:
    void clear()
    {
        std::lock_guard<std::mutex> g(m_);
        for ( auto& v: list_ )
        {
            znn_free(v);
        }
        list_.clear();
    }

public:
    void return_cube( cube<T>* c )
    {
        std::lock_guard<std::mutex> g(m_);
        list_.push_back(c);
    }

public:
    single_size_cube_pool( const vec3i& s )
        : size_(s)
        , list_{}
        , m_{}
    {}

    ~single_size_cube_pool()
    {
        clear();
    }

    std::shared_ptr<cube<T>> get()
    {
        cube<T>* r = nullptr;
        {
            std::lock_guard<std::mutex> g(m_);
            if ( list_.size() > 0 )
            {
                r = list_.back();
                list_.pop_back();
            }
        }

        if ( !r )
        {
            r = malloc_cube<T>(size_);
        }

        return std::shared_ptr<cube<T>>(r,[this](cube<T>* c)
                                        {
                                            this->return_cube(c);
                                        });
    }

};

template<typename T>
class single_size_qube_pool
{
private:
    vec4i                      size_;
    std::list<qube<T>*>        list_;
    std::mutex                 m_   ;

public:
    void clear()
    {
        std::lock_guard<std::mutex> g(m_);
        for ( auto& v: list_ )
        {
            znn_free(v);
        }
        list_.clear();
    }

public:
    void return_qube( qube<T>* q )
    {
        std::lock_guard<std::mutex> g(m_);
        list_.push_back(q);
    }

public:
    single_size_qube_pool( const vec4i& s )
        : size_(s)
        , list_{}
        , m_{}
    {}

    ~single_size_qube_pool()
    {
        clear();
    }

    std::shared_ptr<qube<T>> get()
    {
        qube<T>* r = nullptr;
        {
            std::lock_guard<std::mutex> g(m_);
            if ( list_.size() > 0 )
            {
                r = list_.back();
                list_.pop_back();
            }
        }

        if ( !r )
        {
            r = malloc_qube<T>(size_);
        }

        return std::shared_ptr<cube<T>>(r,[this](qube<T>* c)
                                        {
                                            this->return_qube(c);
                                        });
    }

};


template< typename T >
class single_type_cube_pool
{
private:
    std::mutex                                   m_;
    std::map<vec3i, single_size_cube_pool<T>*>   pools_;

    single_size_cube_pool<T>* get_pool( const vec3i& s )
    {
        std::lock_guard<std::mutex> g(m_);
        if ( pools_.count(s) )
        {
            return pools_[s];
        }
        else
        {
            single_size_cube_pool<T>* r = new single_size_cube_pool<T>{s};
            pools_[s] = r;
            return r;
        }
    }

public:
    std::shared_ptr<cube<T>> get( const vec3i& s )
    {
        return get_pool(s)->get();
    }

    void return_cube( cube<T>* c )
    {
        get_pool(vec3i(c->shape()[0], c->shape()[1], c->shape()[2]))->return_cube(c);
    }

}; // single_type_cube_pool

template< typename T >
class single_type_qube_pool
{
private:
    std::mutex                                   m_;
    std::map<vec4i, single_size_qube_pool<T>*>   pools_;

    single_size_qube_pool<T>* get_pool( const vec4i& s )
    {
        std::lock_guard<std::mutex> g(m_);
        if ( pools_.count(s) )
        {
            return pools_[s];
        }
        else
        {
            single_size_qube_pool<T>* r = new single_size_qube_pool<T>{s};
            pools_[s] = r;
            return r;
        }
    }

public:
    std::shared_ptr<qube<T>> get( const vec4i& s )
    {
        return get_pool(s)->get();
    }

    void return_cube( qube<T>* c )
    {
        get_pool(vec4i(c->shape()[0], c->shape()[1],
                       c->shape()[2], c->shape()[3]))->return_cube(c);
    }

}; // single_type_cube_pool



template< typename T >
struct pool
{
private:
    static single_type_cube_pool<T>& cinstance;
    static single_type_qube_pool<T>& qinstance;

public:
    static std::shared_ptr<cube<T>> get_cube( const vec3i& s )
    {
        return cinstance.get(s);
    }

    static std::shared_ptr<cube<T>> get_cube( size_t x, size_t y, size_t z )
    {
        return cinstance.get( vec3i(x,y,z) );
    }

    static std::shared_ptr<qube<T>> get_qube( const vec4i& s )
    {
        return qinstance.get(s);
    }

    static std::shared_ptr<qube<T>> get_qube( size_t x, size_t y,
                                              size_t z, size_t t )
    {
        return qinstance.get( vec4i(x,y,z,t) );
    }

};

template< typename T >
single_type_cube_pool<T>& pool<T>::cinstance =
    zi::singleton<single_type_cube_pool<T>>::instance();


template< typename T >
single_type_qube_pool<T>& pool<T>::qinstance =
    zi::singleton<single_type_qube_pool<T>>::instance();


template<typename T>
std::shared_ptr<cube<T>> get_cube(const vec3i& s)
{
    return pool<T>::get_cube(s);
}


template<typename T>
std::shared_ptr<qube<T>> get_qube(const vec4i& s)
{
    return pool<T>::get_qube(s);
}


}} // namespace znn::v4
