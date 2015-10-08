#pragma once

#include "../../utils/simple_accumulator.hpp"
#include "edges_fwd.hpp"
#include "nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class softmax_edge: public edge
{
public:

    class layer
    {
    private:
        std::vector<softmax_edge*> edges_;
        simple_accumulator         accum_;
        task_manager &             tm_   ;

    private:
        void attach(size_t n, softmax_edge* e)
        {
            edges_[n] = e;
        }

        void add_to_the_sum(cube_p<real>&& f)
        {
            if ( accum_.add(std::move(f)) )
            {
                auto sum = accum_.reset();
                for ( auto & e: edges_ )
                {
                    tm_.schedule(e->fwd_priority(), [e,sum]() {
                            e->shoot(sum);
                        });
                }
            }
        }

        friend class softmax_edge;

    public:
        layer( size_t n, task_manager & tm )
            : edges_(n)
            , accum_(n)
            , tm_(tm)
        {}
    };


private:
    std::shared_ptr<layer> layer_data;
    cube_p<real>           last_input = nullptr;

public:
    void shoot( ccube_p<real> const & sum )
    {
        *last_input /= *sum;
        out_nodes->forward(out_num,get_copy(*last_input));
    }

    friend class layer;

public:
    softmax_edge( nodes * in,
                  size_t inn,
                  nodes * out,
                  size_t outn,
                  task_manager & tm,
                  std::shared_ptr<layer> const & layer )
        : edge(in,inn,out,outn,tm)
        , layer_data(layer)
    {
        ZI_ASSERT(inn=outn);
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
        layer->attach(inn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        last_input = exp(*f);
        layer_data->add_to_the_sum(get_copy(*last_input));
    }

    void backward( ccube_p<real> const & g ) override
    {

    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};





}}} // namespace znn::v4::parallel_network
