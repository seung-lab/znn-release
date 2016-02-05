#pragma once

#include "../types.hpp"
#include <cstdlib>

namespace znn { namespace fwd {

#if defined(ZNN_MEM_ALIGN)

#else

template<typename T>
inline T * znn_malloc( long_t s )
{
    void* r = std::malloc(s * sizeof(T));
    if ( !r ) throw std::bad_alloc();
    return reinterpret_cast<T*>(r);
}

inline void znn_free( void* m )
{
    std::free(m);
}

#endif

}}
