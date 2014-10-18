//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_ALLOCATOR_HPP_INCLUDED
#define ZNN_ALLOCATOR_HPP_INCLUDED

#include <fftw3.h>
#include <cstddef>

namespace zi {
namespace znn {

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

    allocator() {}
    allocator(const allocator&) {}

    template <class U>
    allocator(const allocator<U>&) {}

    ~allocator() {}

    pointer address(reference x) const
    { return &x; }

    const_pointer address(const_reference x) const
    { return x; }

    pointer allocate(size_type n, const_pointer = 0)
    {
        void* p = fftw_malloc(n * sizeof(T));
        if (!p)
            throw std::bad_alloc();
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type)
    { fftw_free(p); }

    size_type max_size() const
    { return static_cast<size_type>(-1) / sizeof(T); }

    void construct(pointer p, const value_type& x)
    { new(p) value_type(x); }

    void destroy(pointer p)
    { p->~value_type(); }

    void operator=(const allocator&x)
    { }
};

template<> class allocator<void>
{
    typedef void        value_type;
    typedef void*       pointer;
    typedef const void* const_pointer;

    template <class U>
        struct rebind { typedef allocator<U> other; };
};

template <class T>
inline bool operator==(const allocator<T>&,
                       const allocator<T>&) {
    return true;
}

template <class T>
inline bool operator!=(const allocator<T>&,
                       const allocator<T>&) {
    return false;
}

}} // namespace zi::znn

#endif // ZNN_ALLOCATOR_HPP_INCLUDED