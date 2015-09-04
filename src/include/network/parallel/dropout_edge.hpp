#pragma once

#include "edges_fwd.hpp"
#include "nodes.hpp"
#include "../../initializator/bernoulli_init.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class dropout_edge: public edge
{
private:
    real            ratio_; // keeping ratio
    cube_p<bool>    mask_ ; // dropout mask
    vec3i           insize;
    
    phase           phase_; // TRAIN or TEST

private:
    inline real scale() const
    { 
        return static_cast<real>(1)/ratio_;
    }

    // performs inplace dropout and returns dropout mask
    inline void dropout_forward(cube<real>& f)
    {
        if ( !mask_ )
        {
            mask_ = get_cube<bool>(size(f));
        }

        // new random mask
        bernoulli_init<bool>(ratio_).initialize(mask_);

        size_t s = f.num_elements();
        for ( size_t i = 0; i < s; ++i )
        {
            // dropout
            if ( mask_->data()[i] )
                f.data()[i] *= scale();
            else
                f.data()[i]  = static_cast<real>(0);
        }
    }
    
    // performs inplace dropout backward
    inline void dropout_backward(cube<real> & g)
    {
        ZI_ASSERT(mask_);

        size_t s = g.num_elements();
        for ( size_t i = 0; i < s; ++i )
        {
            if ( mask_->data()[i] )
                g.data()[i] *= scale();
            else
                g.data()[i]  = static_cast<real>(0);
        }

        // Should we reset mask_ here?
    }

public:
    dropout_edge( nodes * in,
                  size_t inn,
                  nodes * out,
                  size_t outn,
                  task_manager & tm,
                  real p,
                  phase phs = phase::TRAIN )
        : edge(in,inn,out,outn,tm)
        , ratio_(p)
        , phase_(phs)
    {
        insize = in->fsize();

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        ZI_ASSERT(size(*f)==insize);
        
        auto fmap = get_copy(*f);
        if ( phase_ == phase::TRAIN )
        {
            dropout_forward(*fmap);
        }

        out_nodes->forward(out_num, std::move(fmap));
    }

    void backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(insize==size(*g));

        auto gmap = get_copy(*g);
        if ( phase_ == phase::TRAIN )
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
