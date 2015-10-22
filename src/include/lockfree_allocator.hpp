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
#include <boost/lockfree/stack.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/utility/addressof.hpp>
#include <array>

#ifdef ZNN_XEON_PHI
#  include <mkl.h>
#endif

namespace znn { namespace v4 { namespace detail {

inline void* znn_malloc(size_t s)
{
#ifdef ZNN_XEON_PHI
    void* r = mkl_malloc(s,8);
#else
    void* r = malloc(s);
#endif
    if ( !r ) throw std::bad_alloc();
    return r;
}

inline void znn_free(void* ptr)
{
#ifdef ZNN_XEON_PHI
    mkl_free(ptr);
#else
    free(ptr);
#endif
}


class memory_bucket
{
public:
    std::size_t                   mem_size_;
    boost::lockfree::queue<void*> stack_   ;

public:
    memory_bucket(size_t ms = 0)
        : mem_size_(ms)
        , stack_(65536*4)
    {}

    ~memory_bucket()
    {
        clear();
    }

    void clear()
    {
        void * p;
        while ( stack_.unsynchronized_pop(p) )
        {
            znn_free(p);
        }
    }

public:
    void ret( void* c )
    {
        while ( !stack_.push(c) );
    }


    void* get()
    {
        void* r;
        if ( stack_.pop(r) ) return r;
        return znn_malloc(mem_size_);
    }
};

class bucket_pool_impl
{
private:
    std::array<memory_bucket,32> buckets_;

public:
    bucket_pool_impl()
    {
        for ( size_t i = 0; i < 32; ++i )
        {
            buckets_[i].mem_size_ = static_cast<size_t>(1) << i;
        }
    }

public:
    void* get( std::size_t s )
    {
        size_t bucket = 64 - __builtin_clzl( s - 1 );
        return this->buckets_[bucket].get();
    }

    void ret( void* p, std::size_t s )
    {
        size_t bucket = 64 - __builtin_clzl( s - 1 );
        return this->buckets_[bucket].ret(p);
    }

};

namespace {
bucket_pool_impl& bucket_pool = zi::singleton<bucket_pool_impl>::instance();
}

template <typename T>
class allocator
{
public:
    typedef T                 value_type;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;
    typedef value_type&       reference;
    typedef const value_type& const_reference;
    typedef std::size_t       size_type;
    typedef std::ptrdiff_t    difference_type;

    template <class U>
    struct rebind
    {
        typedef allocator<U> other;
    };

    allocator() noexcept {}
    allocator(const allocator&) {}

    template <class U>
    allocator(const allocator<U>&) noexcept {}

    ~allocator() noexcept {}

    pointer address(reference x) const noexcept
    { return &x; }
    const_pointer address(const_reference x) const noexcept
    { return &x; }
    size_type max_size() const noexcept
    { return (std::numeric_limits<size_type>::max)(); }

    template <class U, class... Args>
    void construct(U* p, Args&&... args)
    { ::new((void *)p) U(std::forward<Args>(args)...); }

    template< class U >
    void destroy( U* p )
    { p->~U(); }

    // bool operator==(const allocator&) const
    // { return true; }
    // bool operator!=(const allocator&) const
    // { return true; }


    pointer allocate( size_type n, std::allocator<void>::const_pointer hint=0 )
    {
        return reinterpret_cast<pointer>(bucket_pool.get( n * sizeof(T) ));
    }

    void deallocate(const pointer ptr, const size_type n)
    {
        bucket_pool.ret( ptr, n * sizeof(T) );
    }
};

template<> class allocator<void>
{
public:

    typedef void        value_type;
    typedef void*       pointer;
    typedef const void* const_pointer;

    template <class U>
        struct rebind { typedef allocator<U> other; };
};


template< class T1, class T2 >
bool operator==( const allocator<T1>&, const allocator<T2>&)
{ return true; }

template< class T1, class T2 >
bool operator!=( const allocator<T1>&, const allocator<T2>&)
{ return false; }

} // namespace detail

using detail::allocator;

}} // namespace znn::v4:
