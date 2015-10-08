#pragma once

#include "../../types.hpp"
#include "../../cube/cube.hpp"

#include "../detail/constant.hpp"
#include "detail/convolve.hpp"
#include "detail/sparse.hpp"

namespace znn { namespace v4 {

template< typename T >
inline cube_p<T> convolve_sparse( cube<T> const & a,
                                  cube<T> const & b,
                                  vec3i   const & s )
{
    if ( b.num_elements() == 1 )
    {
        auto r = get_cube<T>(size(a));
        detail::convolve_constant(a, b.data()[0], *r);
        return r;
    }
    else if ( s == vec3i::one )
    {
        auto r = get_cube<T>(vec3i::one + size(a) - size(b));
        detail::pure_convolve(a, b, *r);
        return r;
    }
    else
    {
        auto r = get_cube<T>(size(a) - (size(b) - vec3i::one)*s);
        detail::convolve_sparse(a,b,s,*r);
        return r;
    }
}

template< typename T >
inline void convolve_sparse_add( cube<T> const & a,
                                 cube<T> const & b,
                                 vec3i   const & s,
                                 cube<T> & r ) noexcept
{
    if ( b.num_elements() == 1 )
    {
        detail::convolve_constant_add(a, b.data()[0], r);
    }
    else if ( s == vec3i::one )
    {
        detail::pure_convolve_add(a, b, r);
    }
    else
    {
        detail::convolve_sparse_add(a,b,s,r);
    }
}

template< typename T >
inline cube_p<T> convolve_sparse_flipped( cube<T> const & a,
                                          cube<T> const & b,
                                          vec3i   const & s )

{
    if ( size(a) == size(b) )
    {
        auto r = get_cube<T>(vec3i::one);
        r->data()[0] = detail::convolve_constant_flipped(a, b);
        return r;
    }
    else if ( s == vec3i::one )
    {
        auto r = get_cube<T>(vec3i::one + size(a) - size(b));
        detail::pure_convolve_flipped(a, b, *r);
        return r;
    }
    else
    {
        auto r = get_cube<T>( (size(a)-size(b)) / s + vec3i::one);
        detail::convolve_sparse_flipped(a,b,s,*r);
        return r;
    }
}

template< typename T >
inline cube_p<T> convolve_sparse_inverse( cube<T> const & a,
                                          cube<T> const & b,
                                          vec3i   const & s )
{
    if ( b.num_elements() == 1 )
    {
        auto r = get_cube<T>(size(a));
        detail::convolve_constant_inverse(a,b.data()[0],*r);
        return r;
    }
    else if ( s == vec3i::one )
    {
        auto r = get_cube<T>(size(a) + size(b) - vec3i::one);
        detail::pure_convolve_inverse(a,b,*r);
        return r;
    }
    else
    {
        auto r = get_cube<T>( size(a) + (size(b) - vec3i::one) * s );
        detail::convolve_sparse_inverse(a,b,s,*r);
        return r;
    }
}

template< typename T >
inline void convolve_sparse_inverse_add( cube<T> const & a,
                                         cube<T> const & b,
                                         vec3i   const & s,
                                         cube<T> & r ) noexcept
{
    if ( b.num_elements() == 1 )
    {
        detail::convolve_constant_add(a,b.data()[0],r);
    }
    else if ( s == vec3i::one )
    {
        detail::pure_convolve_inverse_add(a,b,r);
    }
    else
    {
        detail::convolve_sparse_inverse(a,b,s,r);
    }
}


}} // namespace znn::v4
