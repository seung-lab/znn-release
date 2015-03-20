#pragma once

#include <complex>
#include <memory>
#include <type_traits>

#include <boost/multi_array.hpp>
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


template <typename T>
using cube = boost::multi_array<T, 3, typename std::conditional<
                                          needs_fft_allocator<T>::value,
                                          cube_allocator<T>,
                                          std::allocator<T>>::type >;

template <typename T>
using qube = boost::multi_array<T, 4, typename std::conditional<
                                          needs_fft_allocator<T>::value,
                                          cube_allocator<T>,
                                          std::allocator<T>>::type >;


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

template<typename T>
cube_p<T> get_cube(const vec3i& s)
{
    return cube_p<T>(new cube<T>(extents[s[0]][s[1]][s[2]]));
}

template<typename T>
qube_p<T> get_qube(const vec4i& s)
{
    return qube_p<T>(new qube<T>(extents[s[0]][s[1]][s[2]][s[3]]));
}

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
