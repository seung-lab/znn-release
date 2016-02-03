#pragma once

#include "../types.hpp"
#include "../assert.hpp"

namespace znn { namespace fwd {


template<typename T>
inline void pool_inplace_2( T * v, long_t vs,
                            vec3i const & sz,
                            vec3i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            for ( long_t k = 0, z = y; k < sz[2]; ++k, z += st[2] )
            {
                v[z] = std::max(v[z], v[z+vs]);
            }
        }
    }
}


template<typename T>
inline void pool_inplace_3( T * v, long_t vs,
                            vec3i const & sz,
                            vec3i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            for ( long_t k = 0, z = y; k < sz[2]; ++k, z += st[2] )
            {
                v[z] = std::max(v[z], v[z+vs]);
                v[z] = std::max(v[z], v[z+vs*2]);
            }
        }
    }
}

template<typename T>
inline void pool_inplace_4( T * v, long_t vs,
                            vec3i const & sz,
                            vec3i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            for ( long_t k = 0, z = y; k < sz[2]; ++k, z += st[2] )
            {
                v[z] = std::max(v[z], v[z+vs]);
                v[z] = std::max(v[z], v[z+vs*2]);
                v[z] = std::max(v[z], v[z+vs*3]);
            }
        }
    }
}


template<typename T>
inline void pooling_separation( T const * src,
                                T * dst,
                                vec3i const & istrides,
                                vec3i const & ostrides,
                                vec3i const & size )
{
    for ( long_t i = 0, ix = 0, ox = 0;
          i < size[0];
          ++i, ix += istrides[0], ox += ostrides[0] )
    {
        for ( long_t j = 0, iy = ix, oy = ox;
              j < size[1];
              ++j, iy += istrides[1], oy += ostrides[1] )
        {
            for ( long_t k = 0;
                  k < size[2];
                  ++k )
            {
                dst[oy + k * ostrides[2]] = src[iy + k * istrides[2]];
            }
        }
    }
}




}} // namespace znn::fwd
