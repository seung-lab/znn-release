#pragma once

#ifdef ZNN_USE_MKL_FFT
#  include <fftw/fftw3.h>
#else
#  include <fftw3.h>
#endif

namespace znn { namespace v4 {

template <typename T>
class cube_allocator
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
        typedef cube_allocator<U> other;
    };

    cube_allocator() {}
    cube_allocator(const cube_allocator&) {}

    template <class U>
    cube_allocator(const cube_allocator<U>&) {}

    ~cube_allocator() {}

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

    void operator=(const cube_allocator&)
    { }
};

template<> class cube_allocator<void>
{
    typedef void        value_type;
    typedef void*       pointer;
    typedef const void* const_pointer;

    template <class U>
        struct rebind { typedef cube_allocator<U> other; };
};

template <class T>
inline bool operator==(const cube_allocator<T>&,
                       const cube_allocator<T>&) {
    return true;
}

template <class T>
inline bool operator!=(const cube_allocator<T>&,
                       const cube_allocator<T>&) {
    return false;
}

}} // namespace znn:v4
