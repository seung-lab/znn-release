//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
//                    2016  Kisuk Lee           <kisuklee@mit.edu>
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

#include "edge.hpp"
#include "nodes.hpp"
#include "../../utils/dispatcher.hpp"
#include "../../utils/mult_accumulator.hpp"
#include "../../utils/accumulator.hpp"
#include "../../utils/waiter.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class multiply_nodes: public nodes
{
private:
    dispatcher_group<concurrent_forward_dispatcher<edge,edge>>    fwd_dispatch_;
    dispatcher_group<concurrent_backward_dispatcher<edge,edge>>   bwd_dispatch_;

    std::vector<std::unique_ptr<mult_accumulator>>     fwd_accumulators_;
    std::vector<std::unique_ptr<backward_accumulator>> bwd_accumulators_;

    std::vector<cube_p<real>>    fs_      ;
    std::vector<int>             fwd_done_;
    waiter                       waiter_  ;

    // Princeton descent
    std::vector<real>           means_   ; // feature map means
    std::vector<real>           vars_    ; // feature map variances
    std::vector<int>            norms_   ; // normalize or not

public:
    multiply_nodes( size_t s,
                    vec3i const & fsize,
                    options const & op,
                    task_manager & tm,
                    size_t fwd_p,
                    size_t bwd_p,
                    bool is_out )
        : nodes(s,fsize,op,tm,fwd_p,bwd_p,false,is_out)
        , fwd_dispatch_(s)
        , bwd_dispatch_(s)
        , fwd_accumulators_(s)
        , bwd_accumulators_(s)
        , fs_(s)
        , fwd_done_(s)
        , waiter_(s)
        , means_(s)
        , vars_(s)
        , norms_(s)
    {
        for ( size_t i = 0; i < nodes::size(); ++i )
        {
            fwd_accumulators_[i]
                = std::make_unique<mult_accumulator>();
            bwd_accumulators_[i]
                = std::make_unique<backward_accumulator>(fsize);
        }

        auto type = op.require_as<std::string>("type");
        ZI_ASSERT(type=="multiply");
    }

    options serialize() const override
    {
        options ret = nodes::opts();
        return ret;
    }

public:
    size_t num_out_nodes() override { return nodes::size(); }
    size_t num_in_nodes()  override { return nodes::size(); }

    std::vector<cube_p<real>>& get_featuremaps() override
    {
        return fs_;
    }

    std::vector<real>& get_means() override
    {
        return means_;
    }

    std::vector<real>& get_variances() override
    {
        return vars_;
    }

private:
    void do_forward(size_t n)
    {
        ZI_ASSERT(enabled_[n]);

        fs_[n] = fwd_accumulators_[n]->reset();
        //STRONG_ASSERT(!fwd_done_[n]);
        fwd_done_[n] = true;

        if ( nodes::is_output() )
        {
            waiter_.one_done();
        }
        else
        {
            // Princeton descent
            // TODO(lee):
            //  running average
            means_[n] = mean(*fs_[n]);
            vars_[n]  = variance(*fs_[n]);

            if ( norms_[n] )
            {
                const real epsilon = 1e-5f;

                auto f = get_copy(*fs_[n]);
                *f -= means_[n];
                *f /= vars_[n] + epsilon;

                fwd_dispatch_.dispatch(n,fs_[n],f,nodes::manager());
            }
            else
            {
                fwd_dispatch_.dispatch(n,fs_[n],nodes::manager());
            }
        }
    }

public:
    void forward(size_t n, cube_p<real>&& f)
    {
        ZI_ASSERT(n<nodes::size());
        if ( !enabled_[n] ) return;

        if ( fwd_accumulators_[n]->mult(std::move(f)) )
        {
            do_forward(n);
        }
    }

private:
    void do_backward(size_t n, cube_p<real> const & g)
    {
        ZI_ASSERT(enabled_[n]);

        //STRONG_ASSERT(fwd_done_[n]);
        fwd_done_[n] = false;

        *g *= *fs_[n];

        bwd_dispatch_.dispatch(n,g,nodes::manager());
    }

public:
    void backward(size_t n, cube_p<real>&& g) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !enabled_[n] ) return;

        if ( nodes::is_output() )
        {
            do_backward(n,g);
        }
        else
        {
            if ( bwd_accumulators_[n]->add(std::move(g)) )
            {
                do_backward(n,bwd_accumulators_[n]->reset());
            }
        }
    }

public:
    void attach_out_edge(size_t n, edge* e) override
    {
        ZI_ASSERT(n<nodes::size());

        // Princeton descent
        if ( e->trainable() )
            norms_[n] = true;

        fwd_dispatch_.sign_up(n,e);
        bwd_accumulators_[n]->grow(1);
    }

    void attach_in_edge(size_t n, edge* e) override
    {
        ZI_ASSERT(n<nodes::size());
        bwd_dispatch_.sign_up(n,e);
        fwd_accumulators_[n]->grow(1);
    }

    size_t attach_out_fft_edge(size_t n, edge* e, vec3i const & s) override
    {
        ZI_ASSERT(n<nodes::size());

        // Princeton descent
        if ( e->trainable() )
            norms_[n] = true;

        fwd_dispatch_.sign_up(n,s,e);
        return bwd_accumulators_[n]->grow_fft(s,1);
    }

protected:
    void disable_fwd(size_t n) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !enabled_[n] ) return;

        // diable outgoing edges
        fwd_dispatch_.enable(n,false);
        bwd_accumulators_[n]->enable_all(false);

        // reset feature map
        fs_[n].reset();

        enabled_[n] = false;
        if ( nodes::is_output() )
            waiter_.dec();
    }

    void disable_bwd(size_t n) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !enabled_[n] ) return;

        // disable incoming edges
        bwd_dispatch_.enable(n,false);
        fwd_accumulators_[n]->reset(0);

        // reset feature map
        fs_[n].reset();

        enabled_[n] = false;
        if ( nodes::is_output() )
            waiter_.dec();
    }

public:
    void enable(size_t n, bool b) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( enabled_[n] == b ) return;

        fwd_dispatch_.enable(n,b);
        bwd_accumulators_[n]->enable_all(b);

        bwd_dispatch_.enable(n,b);
        fwd_accumulators_[n]->reset(b ? bwd_dispatch_.size(n) : 0);

        // reset feature map
        fs_[n].reset();

        enabled_[n] = b;
        if ( nodes::is_output() )
            b ? waiter_.inc() : waiter_.dec();
    }

    void enable_out_edge(size_t n, bool b) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !bwd_accumulators_[n]->enable(b) )
            disable_bwd(n);
    }

    void enable_in_edge(size_t n, bool b, bool) override
    {
        ZI_ASSERT(n<nodes::size());
        size_t s = b ? fwd_accumulators_[n]->grow(1)
                     : fwd_accumulators_[n]->shirink(1);
        if ( !s ) disable_fwd(n);
    }

    void enable_out_fft_edge(size_t n, bool b, vec3i const & s) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !bwd_accumulators_[n]->enable_fft(s,b) )
            disable_bwd(n);
    }

public:
    void set_eta( real /*eta*/ ) override {}
    void set_momentum( real /*mom*/ ) override {}
    void set_weight_decay( real /*wd*/ ) override {}

    void wait() override { waiter_.wait(); }

    void zap() override {}

}; // class multiply_nodes

}}} // namespace znn::v4::parallel_network
