#pragma once

#include "edges_fwd.hpp"
#include "nodes.hpp"
#include "../../pooling/pooling.hpp"


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
        insize = in->fsize();
    }

    void forward( ccube_p<double> const & f ) override
    {
        ZI_ASSERT(size(*f)==insize);
        auto r = pooling_filter(get_copy(*f),
                                [](double a, double b){ return a>b; },
                                filter_size,
                                filter_stride);
        indices = r.second;
        out_nodes->forward(out_num,std::move(r.first));
    }

    void backward( ccube_p<double> const & g )
    {
        ZI_ASSERT(indices);
        ZI_ASSERT(insize==size(*g)+(filter_size-vec3i::one)*filter_stride);
        in_nodes->backward(in_num, pooling_backprop(insize, *g, *indices));
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};


}}} // namespace znn::v4::parallel_network
