#pragma once

#include <iostream>
#include "optimal.hpp"
#include "../../assert.hpp"
#include "../../types.hpp"

namespace znn { namespace fwd {

class fft_transformer_base
{
protected:
    vec3i isize ; // image to be convolved size
    vec3i ksize ; // kernel size
    vec3i rsize ; // result size = isize - kernel + 1
    vec3i asize ; // actual transform size (to be padded to)
    vec3i csize ; // transform size

    real scale;

protected:
    fft_transformer_base( vec3i const & is, vec3i const & ks )
        : isize(is)
        , ksize(ks)
        , rsize(is+vec3i::one-ks)
    {
        STRONG_ASSERT(is[0] >= ks[0]);
        STRONG_ASSERT(is[1] >= ks[1]);
        STRONG_ASSERT(is[2] >= ks[2]);

        asize = get_optimal_size(isize);
        csize = asize;
        csize[0] /= 2; csize[0] += 1;
        scale = asize[0] * asize[1] * asize[2];
    }

public:
    bool needs_padding() const
    {
        return isize != asize;
    }

    real get_scale() const
    {
        return scale;
    }

    vec3i const & image_size() const
    {
        return isize;
    }

    vec3i const & kernel_size() const
    {
        return ksize;
    }

    vec3i const & result_size() const
    {
        return rsize;
    }

    vec3i const & actual_size() const
    {
        return asize;
    }

    vec3i const & transform_size() const
    {
        return csize;
    }

    long_t image_elements() const
    {
        return isize[0] * isize[1] * isize[2];
    }

    long_t kernel_elements() const
    {
        return ksize[0] * ksize[1] * ksize[2];
    }

    long_t result_elements() const
    {
        return rsize[0] * rsize[1] * rsize[2];
    }

    long_t image_scratch_elements() const
    {
        return asize[0] * isize[1] * isize[2];
    }

    long_t kernel_scratch_elements() const
    {
        return asize[0] * ksize[1] * ksize[2];
    }

    long_t result_scratch_elements() const
    {
        return asize[0] * rsize[1] * rsize[2];
    }

    long_t actual_elements() const
    {
        return asize[0] * asize[1] * asize[2];
    }

    long_t transform_elements() const
    {
        return csize[0] * csize[1] * csize[2];
    }

    long_t image_memory() const
    {
        return isize[0] * isize[1] * isize[2] * sizeof(real);
    }

    long_t kernel_memory() const
    {
        return ksize[0] * ksize[1] * ksize[2] * sizeof(real);
    }

    long_t result_memory() const
    {
        return rsize[0] * rsize[1] * rsize[2] * sizeof(real);
    }

    long_t image_scratch_memory() const
    {
        return asize[0] * isize[1] * isize[2] * sizeof(real);
    }

    long_t kernel_scratch_memory() const
    {
        return asize[0] * ksize[1] * ksize[2] * sizeof(real);
    }

    long_t result_scratch_memory() const
    {
        return asize[0] * rsize[1] * rsize[2] * sizeof(real);
    }

    long_t transform_memory() const
    {
        return csize[0] * csize[1] * csize[2] * sizeof(real) * 2;
    }
};

}} // namespace znn::fwd
