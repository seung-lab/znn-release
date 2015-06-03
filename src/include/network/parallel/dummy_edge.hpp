#pragma once

#include "edges_fwd.hpp"
#include "nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class dummy_edge: public edge
{
private:
    vec3i filter_size;
    vec3i filter_stride;

    cube_p<int> indices;
    vec3i       insize ;

public:
    dummy_edge( nodes * in,
                size_t inn,
                nodes * out,
                size_t outn,
                task_manager & tm )
        : edge(in,inn,out,outn,tm)
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        out_nodes->forward(out_num,get_copy(*f));
    }

    void backward( ccube_p<real> const & g )
    {
        in_nodes->backward(in_num,get_copy(*g));
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};


}}} // namespace znn::v4::parallel_network
