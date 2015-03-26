#pragma once

#include <cstdint>
#include <cstddef>
#include <complex>
#include <mutex>

#include <zi/vl/vl.hpp>

namespace znn { namespace v4 {

typedef double                 real;
typedef std::complex<double>   cplx;
typedef std::complex<double>   complex;

typedef zi::vl::vec<int64_t,3> vec3i;
typedef zi::vl::vec<int64_t,3> vec4i;


typedef std::size_t size_t;

typedef std::lock_guard<std::mutex> guard;

typedef int64_t long_t;

}} // namespace znn::v4
