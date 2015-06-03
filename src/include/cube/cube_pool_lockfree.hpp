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
    explicit qube(const vec4i& s, T* data)
        : boost::multi_array_ref<T,4>(data,extents[s[0]][s[1]][s[2]][s[3]])
    {
    }

    ~qube()
    {
        znn_free(this);
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
class memory_bucket
{
public:
    std::size_t                   mem_size_;
    boost::lockfree::stack<void*> stack_   ;

public:
    memory_bucket(size_t ms = 0)
        : mem_size_(ms)
    {}

    void clear()
    {
        void * p;
        while ( stack_.unsynchronized_pop(p) )
        {
            znn_free(p);
        }
    }

public:
    void return_memory( void* c )
    {
        while ( !stack_.push(c) );
    }

public:
    ~memory_bucket()
    {
        clear();
    }

    void* get()
    {
        void* r;
        if ( stack_.pop(r) ) return r;
        return znn_malloc(mem_size_);
    }
};

template< typename T >
class single_type_cube_pool
{
private:
    std::array<memory_bucket<T>,48> buckets_;

public:
    single_type_cube_pool()
    {
        for ( size_t i = 0; i < 48; ++i )
        {
            buckets_[i].mem_size_ = static_cast<size_t>(1) << i;
        }
    }

public:
    std::shared_ptr<cube<T>> get( const vec3i& s )
    {
        size_t bucket = 64 - __builtin_clzl( __znn_aligned_size<cube<T>>::value
                                             + s[0]*s[1]*s[2]*sizeof(T) - 1 );

        void*    mem  = buckets_[bucket].get();
        T*       data = __offset_cast<T>(mem, __znn_aligned_size<cube<T>>::value);
        cube<T>* c    = new (mem) cube<T>(s,data);

        return std::shared_ptr<cube<T>>(c,[this,bucket](cube<T>* c) {
                this->buckets_[bucket].return_memory(c);
            });
    }

}; // single_type_cube_pool


template< typename T >
struct pool
{
private:
    static single_type_cube_pool<T>& instance;

public:
    static std::shared_ptr<cube<T>> get( const vec3i& s )
    {
        return instance.get(s);
    }

    static std::shared_ptr<cube<T>> get( size_t x, size_t y, size_t z )
    {
        return instance.get( vec3i(x,y,z) );
    }
};

template< typename T >
single_type_cube_pool<T>& pool<T>::instance =
    zi::singleton<single_type_cube_pool<T>>::instance();


template<typename T>
std::shared_ptr<cube<T>> get_cube(const vec3i& s)
{
    return pool<T>::get(s);
}
