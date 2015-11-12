//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
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
            accum_.inc();
        }

        void detach(size_t n)
        {
            ZI_ASSERT(n<edges_.size());
            edges_[n] = nullptr;
            accum_.dec();
        }

        void add_to_the_sum(cube_p<real>&& f)
        {
            if ( accum_.add(std::move(f)) )
            {
                auto sum = accum_.reset();
                for ( auto & e: edges_ )
                {
                    if ( e )
                        tm_.schedule(e->fwd_priority(),
                                    [e,sum](){e->shoot(sum);});
                }
            }
        }

        friend class softmax_edge;

    public:
        layer( size_t n, task_manager & tm )
            : edges_(n)
            , accum_(0)
            , tm_(tm)
        {}
    };


private:
    std::shared_ptr<layer> layer_data;
    cube_p<real>           last_input = nullptr;

public:
    void shoot( ccube_p<real> const & sum )
    {
        ZI_ASSERT(enabled_);
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
        ZI_ASSERT(inn==outn);
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
        layer->attach(inn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        if ( !enabled_ ) return;

        last_input = exp(*f);
        layer_data->add_to_the_sum(get_copy(*last_input));
    }

    void backward( ccube_p<real> const & g ) override
    {

    }

    void enable_fwd(bool b) override
    {
        if ( enabled_ == b ) return;

        edge::enable_fwd(b);

        if ( enabled_ )
            layer_data->attach(in_num,this);
        else
            layer_data->detach(in_num);
    }

    void enable_bwd(bool b) override
    {
        if ( enabled_ == b ) return;

        edge::enable_bwd(b);

        if ( enabled_ )
            layer_data->attach(in_num,this);
        else
            layer_data->detach(in_num);
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};





}}} // namespace znn::v4::parallel_network
