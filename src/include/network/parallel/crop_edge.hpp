#pragma once

#include "edges_fwd.hpp"
#include "nodes.hpp"


namespace znn { namespace v4 { namespace parallel_network {

class crop_edge: public edge
{
private:
    real            ratio ;
    cube_p<bool>    mask  ;
    vec3i           insize;

private:
    // performs inplace dropout and returns dropout mask
    inline cube_p<bool> dropout(cube_p<real> & f, real p);

public:
    crop_edge( nodes * in,
                  size_t inn,
                  nodes * out,
                  size_t outn,
                  task_manager & tm,
                  real p )
        : edge(in,inn,out,outn,tm)
        , ratio(p)
    {
        insize = in->fsize();

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        ZI_ASSERT(size(*f)==insize);
        
        cube_p<real> fmap = get_copy(*f);
        if ( TRAINING )
        {
            mask = dropout(*fmap, ratio);
        }

        out_nodes->forward(out_num,std::move(fmap));
    }

    void backward( ccube_p<real> const & g )
    {
        // ZI_ASSERT(mask);
        // ZI_ASSERT(insize==size(*g));
        
        // if ( in_nodes->is_input() )
        // {
        //     in_nodes->backward(in_num, cube_p<real>());
        // }
        // else
        // {
        //     in_nodes->backward(in_num, pooling_backprop(insize, *g, *indices));
        // }
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};


}}} // namespace znn::v4::parallel_network
