#pragma once

#include "edges_fwd.hpp"
#include "nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class crop_edge: public edge
{
private:
    vec3i   offset;
    vec3i   insize;

private:
    inline vec3i crop_size() const
    {
        return insize - offset - offset;
    }

public:
    crop_edge( nodes * in,
               size_t inn,
               nodes * out,
               size_t outn,
               task_manager & tm,
               vec3i const & off )
        : edge(in,inn,out,outn,tm)
        , offset(off)
    {
        insize = in->fsize();

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        ZI_ASSERT(size(*f)==insize);
        out_nodes->forward(out_num, crop(*f,offset,crop_size()));
    }

    void backward( ccube_p<real> const & g )
    {        
        ZI_ASSERT(insize==size(*g));
        
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            auto gmap = get_cube<real>(insize);
            in_nodes->backward(in_num, pad_zeros(*g,offset,"both"));
        }
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};

}}} // namespace znn::v4::parallel_network
