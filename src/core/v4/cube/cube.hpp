#pragma once

#include <complex>
#include <memory>
#include <type_traits>
#include <array>

#include <boost/multi_array.hpp>
#include <boost/lockfree/stack.hpp>
#include <zi/utility/singleton.hpp>
#include <jemalloc/jemalloc.h>


#include "cube_allocator.hpp"
#include "../types.hpp"

namespace znn { namespace v4 {

typedef boost::multi_array_types::index_range range;

namespace { decltype(boost::indices) indices; }
namespace { decltype(boost::extents) extents; }

template <typename T>
struct needs_fft_allocator: std::is_floating_point<T> {};

template <typename T>
struct needs_fft_allocator<std::complex<T>>: std::is_floating_point<T> {};


#ifdef DUMMY_CUBE
#  include "dummy_cube.hpp"
#elif defined ZNN_ARENA_CUBE
#  include "znn_arena_cube.hpp"
#else
#  include "znn_malloc_cube.hpp"
#endif


#ifdef NDEBUG

template <typename T> using ccube = cube<T>;
template <typename T> using cqube = qube<T>;

#else

template <typename T> using ccube = const cube<T>;
template <typename T> using cqube = const qube<T>;

#endif

template <typename T> using cube_p  = std::shared_ptr<cube<T>>;
template <typename T> using ccube_p = std::shared_ptr<ccube<T>>;

template <typename T> using qube_p  = std::shared_ptr<qube<T>>;
template <typename T> using cqube_p = std::shared_ptr<cqube<T>>;


// TODO: to be replaced with an efficient cube arena
//       probaby one using jemalloc

template <typename T>
inline vec3i size( const cube<T>& a )
{
    return vec3i(a.shape()[0],a.shape()[1],a.shape()[2]);
};

template <typename T>
inline vec4i size( const qube<T>& a )
{
    return vec4i(a.shape()[0],a.shape()[1],a.shape()[2],a.shape()[3]);
};


}} // namespace znn::v4
