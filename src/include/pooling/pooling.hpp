#pragma once


#include "utils.hpp"

namespace znn { namespace v4 {


template<typename F>
inline void inplace_pooling_filter( cube<real> & featuremap,
                                    cube<int>    & indices,
                                    F const & f,
                                    vec3i const & filter_size,
                                    vec3i const & filter_stride = vec3i::one )
{
    vec3i s = size(featuremap);

    // x-direction
    // delta x is then s[1]*s[2]
    if ( filter_size[0] > 1 )
    {
        for ( long_t y = 0; y < s[1]; ++y )
            for ( long_t z = 0; z < s[2]; ++z )
                for ( long_t x = 0; x < filter_stride[0]; ++x )
                    pooling_filter_pass( &(featuremap[x][y][z]),
                                         &(featuremap[s[0]-1][y][z]),
                                         &(indices[x][y][z]),
                                         filter_size[0],
                                         s[1]*s[2]*filter_stride[0],
                                         f);


    }

    // y-direction
    // delta y is then s[2]
    if ( filter_size[1] > 1 )
    {
        for ( long_t x = 0; x < s[0]; ++x )
            for ( long_t y = 0; y < filter_stride[1]; ++y )
                for ( long_t z = 0; z < s[2]; ++z )
                    pooling_filter_pass( &(featuremap[x][y][z]),
                                         &(featuremap[x][s[1]-1][z]),
                                         &(indices[x][y][z]),
                                         filter_size[1],
                                         s[2]*filter_stride[1],
                                         f);
    }

    // z-direction
    // delta z is 1
    if ( filter_size[2] > 1 )
    {
        for ( long_t x = 0; x < s[0]; ++x )
            for ( long_t y = 0; y < s[1]; ++y )
                for ( long_t z = 0; z < filter_stride[2]; ++z )
                    pooling_filter_pass( &(featuremap[x][y][z]),
                                         &(featuremap[x][y][s[2]-1]),
                                         &(indices[x][y][z]),
                                         filter_size[2],
                                         filter_stride[2],
                                         f);
    }

}

template<typename F>
inline void inplace_pooling_filter_no_indices( cube<real> & featuremap,
                                               F const & f,
                                               vec3i const & filter_size,
                                               vec3i const & filter_stride = vec3i::one )
{
    vec3i s = size(featuremap);

    // x-direction
    // delta x is then s[1]*s[2]
    if ( filter_size[0] > 1 )
    {
        for ( long_t y = 0; y < s[1]; ++y )
            for ( long_t z = 0; z < s[2]; ++z )
                for ( long_t x = 0; x < filter_stride[0]; ++x )
                    pooling_filter_pass_no_indices( &(featuremap[x][y][z]),
                                                    &(featuremap[s[0]-1][y][z]),
                                                    filter_size[0],
                                                    s[1]*s[2]*filter_stride[0],
                                                    f);


    }

    // y-direction
    // delta y is then s[2]
    if ( filter_size[1] > 1 )
    {
        for ( long_t x = 0; x < s[0]; ++x )
            for ( long_t y = 0; y < filter_stride[1]; ++y )
                for ( long_t z = 0; z < s[2]; ++z )
                    pooling_filter_pass_no_indices( &(featuremap[x][y][z]),
                                                    &(featuremap[x][s[1]-1][z]),
                                                    filter_size[1],
                                                    s[2]*filter_stride[1],
                                                    f);
    }

    // z-direction
    // delta z is 1
    if ( filter_size[2] > 1 )
    {
        for ( long_t x = 0; x < s[0]; ++x )
            for ( long_t y = 0; y < s[1]; ++y )
                for ( long_t z = 0; z < filter_stride[2]; ++z )
                    pooling_filter_pass_no_indices( &(featuremap[x][y][z]),
                                                    &(featuremap[x][y][s[2]-1]),
                                                    filter_size[2],
                                                    filter_stride[2],
                                                    f);
    }

}


template<typename F>
inline std::pair<cube_p<real>, cube_p<int>>
pooling_filter( cube_p<real>&& featuremap,
                F const & f,
                vec3i const & filter_size,
                vec3i const & filter_stride = vec3i::one )
{
    auto indices = make_indices(size(*featuremap));
    inplace_pooling_filter(*featuremap, *indices, f,
                           filter_size, filter_stride);

    // the real filter size equals to
    // (SIZE-1) * STRIDE + 1
    // the output is then
    // FMAP_SIZE - REAL_FILTER_SIZE + 1
    // = FMAP_SIZE - (SIZE-1) * STRIDE

    vec3i out_size
        = size(*featuremap) - (filter_size-vec3i::one) * filter_stride;

    return { crop(*featuremap,out_size), crop(*indices,out_size) };

}


template<typename F>
inline cube_p<real>
pooling_filter_no_indices( cube_p<real>&& featuremap,
                           F const & f,
                           vec3i const & filter_size,
                           vec3i const & filter_stride = vec3i::one )
{
    inplace_pooling_filter_no_indices(*featuremap, f,
                                      filter_size, filter_stride);

    // the real filter size equals to
    // (SIZE-1) * STRIDE + 1
    // the output is then
    // FMAP_SIZE - REAL_FILTER_SIZE + 1
    // = FMAP_SIZE - (SIZE-1) * STRIDE

    vec3i out_size
        = size(*featuremap) - (filter_size-vec3i::one) * filter_stride;

    return crop(*featuremap,out_size);

}


inline cube_p<real> pooling_backprop( vec3i         const & sz,
                                        ccube<real> const & vals,
                                        ccube<int>    const & indices )
{
    ZI_ASSERT(size(vals)==size(indices));

    auto ret = get_cube<real>(sz);
    fill(*ret,0);

    real*       rp = ret->data();
    const real* vp = vals.data();
    const int*    ip = indices.data();

    for ( size_t i = 0; i < vals.num_elements(); ++i ) rp[ip[i]] += vp[i];

    return ret;
}




}} // namespace znn:v4
