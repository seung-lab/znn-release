#pragma once

#include "edges_fwd.hpp"
#include "nodes.hpp"
#include "network.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class dropout_edge: public edge
{
private:
    real            ratio ; // dropout ratio
    cube_p<bool>    mask  ; // dropout mask
    vec3i           insize;
    
    network::phase  phase ; // TRAIN or TEST

private:
    inline real scale() const
    { 
        return static_cast<real>(1)/(static_cast<real>(1) - ratio);
    }

    // performs inplace dropout and returns dropout mask
    inline void dropout_forward(cube<real>& f)
    {
        size_t s = f->num_elements();

        if ( !mask ) mask = get_cube<bool>(s);

        for ( size_t i = 0; i < s; ++i )
        {
            // dropout
            if ( zero_one_generator.rand() < ratio )
            {
                f->data()[i]    = static_cast<real>(0);
                mask->data()[i] = false;
            }
            else
            {
                f->data()[i]   *= scale();
                mask->data()[i] = true;
            }
        }
    }
    
    // performs inplace dropout backward
    inline void dropout_backward(cube<real> & g)
    {
        ZI_ASSERT(mask);

        size_t s = g->num_elements();
        for ( size_t i = 0; i < s; ++i )
        {
            if ( mask->data()[i] )
                g->data()[i] *= scale();
            else
                g->data()[i]  = static_cast<real>(0);
        }

        // Should I reset mask here?
    }

public:
    dropout_edge( nodes * in,
                  size_t inn,
                  nodes * out,
                  size_t outn,
                  task_manager & tm,
                  real p,
                  network::phase phs = network::phase::TRAIN )
        : edge(in,inn,out,outn,tm)
        , ratio(p)
        , phase(phs)
    {
        insize = in->fsize();

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        ZI_ASSERT(size(*f)==insize);
        
        auto fmap = get_copy(*f);
        if ( phase == network::phase::TRAIN )
        {
            mask = dropout_forward(*fmap);
        }

        out_nodes->forward(out_num, std::move(fmap));
    }

    void backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(insize==size(*g));

        auto gmap = get_copy(*g);
        if ( phase == network::phase::TRAIN )
        {
            dropout_backward(*gmap);
        }
        
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num, std::move(gmap));
        }
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};


}}} // namespace znn::v4::parallel_network
