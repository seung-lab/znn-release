//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_CORE_CONVOLUTION_FFT_CONVOLVE_HPP_INCLUDED
#define ZNN_CORE_CONVOLUTION_FFT_CONVOLVE_HPP_INCLUDED

#include "../types.hpp"
#include "../volume_utils.hpp"
#include "../utils.hpp"
#include "../fft/fftw.hpp"

namespace zi {
namespace znn {

template< typename T >
inline vol_p<T> fft_convolve( const vol<T>& a, const vol<T>& b)
{
    vec3i as = size(a);

    auto at = fftw::forward(a);
    auto bt = fftw::forward_pad(b, as);

    (*at) *= *bt;

    auto r = fftw::backward(at, as);
    real n = a.num_elements();

    vec3i rs = as + vec3i::one - size(b);

    r = volume_utils::crop_right(r, rs);
    *r /= n;
    return r;
}

template< typename T >
inline vol_p<T> fft_convolve( const cvol_p<T>& a, const cvol_p<T>& b)
{
    return fft_convolve(*a, *b);
}

template< typename T >
inline vol_p<T> fft_convolve_sparse( const vol<T>& a, const vol<T>& b,
                                     const vec3i& s )
{
    vec3i as = size(a);
    auto b2 = sparse_explode(b, s, as);

    auto at = fftw::forward(a);
    auto bt = fftw::forward(*b2);

    (*at) *= *bt;

    auto r = fftw::backward(at, as);
    real n = a.num_elements();

    vec3i rs = as - s * (size(b) - vec3i::one);

    r = volume_utils::crop_right(r, rs);
    *r /= n;
    return r;
}

template< typename T >
inline vol_p<T> fft_convolve_sparse( const cvol_p<T>& a, const cvol_p<T>& b,
                                     const vec3i& s)
{
    return fft_convolve(*a, *b, s);
}


template< typename T >
inline vol_p<T> fft_convolve_flipped( const vol<T>& a, const vol<T>& b)
{
    vec3i as = size(a);
    vec3i bs = size(b);
    vec3i rs = as + vec3i::one - bs;

    // get flipped b
    vol_p<T> bf = get_volume<T>(bs);
    *bf = b;
    flipdims(*bf);

    auto at = fftw::forward(a);
    auto bt = fftw::forward_pad(*bf, as);

    (*at) *= *bt;

    auto r = fftw::backward(at, as);
    real n = a.num_elements();

    r = volume_utils::crop_right(r, rs);
    *r /= n;
    flipdims(*r);
    return r;
}

template< typename T >
inline vol_p<T> fft_convolve_flipped( const cvol_p<T>& a, const cvol_p<T>& b)
{
    return fft_convolve_flipped(*a, *b);
}

template< typename T >
inline vol_p<T> fft_convolve_sparse_flipped( const vol<T>&, const vol<T>&,
                                             const vec3i& )
{
    return vol_p<T>();
}



template< typename T >
inline vol_p<T>
fft_convolve_sparse_flipped( const cvol_p<T>& a, const cvol_p<T>& b,
                             const vec3i& s )
{
    return fft_convolve_sparse_flipped(*a, *b, s);
}


template< typename T >
inline vol_p<T> fft_convolve_inverse( const vol<T>& a, const vol<T>& b)
{
    vec3i as = size(a);
    vec3i bs = size(b);
    vec3i rs = as + bs - vec3i::one;

    // get flipped b
    vol_p<T> bf = get_volume<T>(bs);
    *bf = b;
    flipdims(*bf);

    auto at = fftw::forward_pad(a, rs);
    auto bt = fftw::forward_pad(*bf, rs);

    (*at) *= *bt;

    auto r = fftw::backward(at, rs);
    real n = r->num_elements();

    *r /= n;
    return r;
}

template< typename T >
inline vol_p<T> fft_convolve_inverse( const cvol_p<T>& a, const cvol_p<T>& b)
{
    return fft_convolve_flipped(*a, *b);
}





}} // namespace zi::znn

#endif // ZNN_CORE_CONVOLUTION_FFT_CONVOLVE_HPP_INCLUDED
