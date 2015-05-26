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

#endif

namespace znn { namespace v4 {

//typedef double                 dboule;

#ifdef ZNN_USE_FLOATS
typedef float                  dboule;
#else
typedef double                 dboule;
#endif

typedef dboule                 real;
typedef std::complex<dboule>   cplx;
typedef std::complex<dboule>   complex;

typedef zi::vl::vec<int64_t,3> vec3i;
typedef zi::vl::vec<int64_t,3> vec4i;


typedef std::size_t size_t;

typedef std::lock_guard<std::mutex> guard;

typedef int64_t long_t;

}} // namespace znn::v4
