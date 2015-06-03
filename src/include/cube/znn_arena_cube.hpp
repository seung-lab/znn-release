#define __ZNN_ALIGN 0xF // 16 byte alignment

class __znn_memory_arena_impl
{
private:
    std::array<boost::lockfree::stack<void*>, 100000> buckets_;

public:
    void* alloc(size_t s)
    {
        s += 15; // + 16 - 1
        s |= 0x7FFF;
        s += 1;
        s >>= 15;

        void* r = nullptr;

        if ( s < 100000 )
        {
            if ( buckets_[s].pop(r) ) return r;
        }

        r = je_malloc(s<<15);
        *(reinterpret_cast<size_t*>(r)) = s;
        return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(r)+16);
    }

    void free(void* ptr)
    {
        size_t s = *(reinterpret_cast<size_t*>
                     (reinterpret_cast<uint8_t*>(ptr)-16));

        if ( s < 100000 )
        {
            buckets_[s].push(ptr);
        }
        else
        {
            return je_free(ptr);
        }
    }
};

namespace {
__znn_memory_arena_impl& __znn_memory_arena
    = zi::singleton<__znn_memory_arena_impl>::instance();
}

inline void* znn_malloc(size_t s)
{
    return __znn_memory_arena.alloc(s);
}

inline void znn_free(void* ptr)
{
    __znn_memory_arena.free(ptr);
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
std::shared_ptr<cube<T>> get_qube(const vec4i& s)
{
    void*    mem  = znn_aligned_malloc(__znn_aligned_size<qube<T>>::value
                                       + s[0]*s[1]*s[2]*s[3]*sizeof(T));

    ZI_ASSERT((reinterpret_cast<size_t>(mem)&__ZNN_ALIGN)==0);

    T*       data = __offset_cast<T>(mem,__znn_aligned_size<qube<T>>::value);
    qube<T>* c    = new (mem) qube<T>(s,data);

    return std::shared_ptr<qube<T>>(c,znn_free);
}
