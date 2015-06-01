#pragma once

#include "edge.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class max_pooling_edge: public edge
{
private:
    vec3i filter_size;
    vec3i filter_stride;

    cube_p<int> indices;
    vec3i       insize ;

public:
    max_pooling_edge( nodes * in,
                      size_t inn,
                      nodes * out,
                      size_t outn,
                      task_manager & tm,
                      vec3i const & size,
                      vec3i const & stride )
        : edge(in,inn,out,outn,tm)
        , filter_size(size)
        , filter_stride(stride)
    {
    }

    void forward( ccube_p<real> const & f ) override
    {
        insize = size(*f);
        auto r = pooling_filter(get_copy(*f),
                                [](real a, real b){ return a>b; },
                                filter_size,
                                filter_stride);
        indices = r.second;
        out_nodes->forward(out_num,std::move(r));
    }

    void backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(indices);
        ZI_ASSERT(insize==size(*g)+(filter_size-vec3i::one)*filter_stride);

        in_nodes->backward(in_num, pooling_backprop(insize, *g, *indices));
    }
};


}}} // namespace znn::v4::parallel_network
