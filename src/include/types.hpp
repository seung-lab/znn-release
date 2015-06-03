#pragma once

#include <cstdint>
#include <cstddef>
#include <complex>
#include <mutex>
#include <memory>
#include <zi/vl/vl.hpp>

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
typedef zi::vl::vec<int64_t,3> vec4i;


typedef std::size_t size_t;

typedef std::lock_guard<std::mutex> guard;

typedef int64_t long_t;

}} // namespace znn::v4
