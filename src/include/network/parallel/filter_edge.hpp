#pragma once

#include "edge.hpp"
#include "edges_fwd.hpp"
#include "nodes.hpp"

#include "../../convolution/convolution.hpp"
#include "../filter.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class filter_edge: public edge
{
private:
    vec3i    filter_stride;
    filter & filter_;

    ccube_p<real> last_input;
    ccube_p<real> grad;

    std::mutex m;

private:

    void actual_forward()
    {
        out_nodes->forward(out_num,
                           convolve_sparse(*last_input,
                                           filter_.W(),
                                           filter_stride));
    }

    void actual_update( ccube_p<real> const & f, ccube_p<real> const & g )
    {
        auto dEdW = convolve_sparse_flipped(*f, *g, filter_stride);
        filter_.update(*dEdW);
    }


private:
    void do_forward( ccube_p<real> const & fnext )
    {
        ccube_p<real> g;
        {
            // SCHEDULED f=1, g=1
            guard gg(m);
            if ( grad ) // not done
            {
                if ( last_input )
                    g = std::move(grad);
                else
                    last_input = fnext;
            }
            else // done
            {
                last_input = fnext;
                actual_forward();
            }
        }

        if ( g )
        {
            last_input = fnext;
            actual_update(last_input, g);
            actual_forward();
        }
    }

    void do_update()
    {
        ccube_p<real> f;
        {
            guard gg(m);
            if ( grad ) f = std::move(last_input);
        }

        // if there's g that means update has not been executed
        if ( f )
        {
            actual_update(f, grad);
            {
                guard gg(m);
                grad.reset();
                if ( last_input ) actual_forward();
            }
        }
    }

public:
    filter_edge( nodes * in,
                 size_t inn,
                 nodes * out,
                 size_t outn,
                 task_manager & tm,
                 vec3i const & stride,
                 filter & f )
        : edge(in,inn,out,outn,tm), filter_stride(stride), filter_(f)
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        manager.schedule(this->fwd_priority() * 1024,
                         &filter_edge::do_forward, this, f);
    }

    void backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(last_input);
        grad = g;

        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num,
                               convolve_sparse_inverse(*g,
                                                       filter_.W(),
                                                       filter_stride));
        }
        manager.schedule( this->bwd_priority()/1024, //this->fwd_priority() + 512,
                          &filter_edge::do_update, this );
    }

};


}}} // namespace znn::v4::parallel_network
