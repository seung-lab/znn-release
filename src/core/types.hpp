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

#ifndef ZNN_TYPES_HPP_INCLUDED
#define ZNN_TYPES_HPP_INCLUDED

#include <complex>
#include <mutex>
#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
#include <zi/vl/vl.hpp>
#include <zi/cstdint.hpp>

#include <type_traits>

#include "meta.hpp"
#include "allocator.hpp"


namespace zi {
namespace znn {

typedef std::lock_guard<std::mutex> mutex_guard;

typedef boost::multi_array<double, 3, allocator< double > >    double3d ;
typedef boost::multi_array<std::complex<double>, 3,
                           allocator< std::complex<double> > > complex3d;

typedef boost::shared_ptr<double3d>   double3d_ptr ;
typedef boost::shared_ptr<complex3d>  complex3d_ptr;

// std::allocator is more efficient for basic data structs that will not
// require fft operations

typedef boost::multi_array<bool, 3, std::allocator< bool > > bool3d;
typedef boost::shared_ptr<bool3d>                            bool3d_ptr;

typedef boost::multi_array<int64_t, 3, std::allocator< int64_t > > long3d;
typedef boost::shared_ptr<long3d>                                  long3d_ptr;

typedef zi::vl::vec<std::size_t,3> vec3i;

typedef boost::multi_array_types::index_range range;

namespace {
decltype(boost::indices) indices;
}

typedef int64_t long_t;

typedef std::complex<double> complex;

template <typename T>
struct is_float: std::is_floating_point<T> {};

template <typename T>
struct is_float<std::complex<T>>: std::is_floating_point<T> {};

template <typename T>
using vol = boost::multi_array<T, 3, typename std::conditional<
                                         is_float<T>::value,
                                         allocator<T>,
                                         std::allocator<T>>::type >;

#ifdef NDEBUG

template <typename T>
using cvol = vol<T>;

#else

template <typename T>
using cvol = const vol<T>;

#endif

template <typename T>
using vol_p = boost::shared_ptr<vol<T>>;

template <typename T>
using cvol_p = boost::shared_ptr<cvol<T>>;

typedef std::size_t size_t;

}} // namespace zi::znn

#endif // ZNN_TYPES_HPP_INCLUDED
