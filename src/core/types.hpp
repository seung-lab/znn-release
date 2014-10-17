#ifndef ZNN_TYPES_HPP_INCLUDED
#define ZNN_TYPES_HPP_INCLUDED

#include <complex>
#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
#include <zi/vl/vl.hpp>
#include <zi/cstdint.hpp>

#include "allocator.hpp"


namespace zi {
namespace znn {

typedef boost::multi_array<double, 3, allocator< double > >                             double3d ;
typedef boost::multi_array<std::complex<double>, 3, allocator< std::complex<double> > > complex3d;

typedef boost::shared_ptr<double3d>   double3d_ptr ;
typedef boost::shared_ptr<complex3d>  complex3d_ptr;

// [07/08/2013 kisuklee]
// For mask
// [zlateski]
// std::allocator is more efficient for basic data structs that will not
// require fft operations

typedef boost::multi_array<bool, 3, std::allocator< bool > > bool3d;
typedef boost::shared_ptr<bool3d>                            bool3d_ptr;

typedef boost::multi_array<int64_t, 3, std::allocator< int64_t > > long3d;
typedef boost::shared_ptr<long3d>                                  long3d_ptr;

typedef zi::vl::vec<std::size_t,3> vec3i;

typedef boost::multi_array_types::index_range range;

typedef int64_t long_t;

}} // namespace zi::znn

#endif // ZNN_TYPES_HPP_INCLUDED
