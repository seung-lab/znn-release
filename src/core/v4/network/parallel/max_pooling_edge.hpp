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

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        ZI_ASSERT(size(*f)==insize);
        auto r = pooling_filter(get_copy(*f),
                                [](real a, real b){ return a>b; },
                                filter_size,
                                filter_stride);
        indices = r.second;
        out_nodes->forward(out_num,std::move(r.first));
    }

    void backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(indices);
        ZI_ASSERT(insize==size(*g)+(filter_size-vec3i::one)*filter_stride);
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num, pooling_backprop(insize, *g, *indices));
        }
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};


class real_pooling_edge: public edge
{
private:
    vec3i filter_size;

    cube_p<int> indices;
    vec3i       insize ;

    vec3i       outsize ;

public:
    real_pooling_edge( nodes * in,
                       size_t inn,
                       nodes * out,
                       size_t outn,
                       task_manager & tm,
                       vec3i const & size )
        : edge(in,inn,out,outn,tm)
        , filter_size(size)
    {
        insize = in->fsize();
        outsize = insize / size;

        ZI_ASSERT((insize%size)==vec3i::zero);

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        ZI_ASSERT(size(*f)==insize);
        auto r = pooling_filter(get_copy(*f),
                                [](real a, real b){ return a>b; },
                                filter_size,
                                vec3i::one);

        indices = sparse_implode_slow(*r.second,filter_size,outsize);
        out_nodes->forward(out_num,
                           sparse_implode_slow(*r.first,filter_size,outsize));
    }

    void backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(indices);
        ZI_ASSERT(insize==size(*g)*filter_size);
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num, pooling_backprop(insize, *g, *indices));
        }
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};


}}} // namespace znn::v4::parallel_network
