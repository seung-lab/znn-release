//
// Copyright (C) 2015-2015  Aleksandar Zlateski <zlateski@mit.edu>
//                    2015  Kisuk Lee           <kisuklee@mit.edu>
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
#include "../../utils/max_accumulator.hpp"
#include "../../utils/accumulator.hpp"
#include "../../utils/waiter.hpp"
#include "../trivial/utils.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class maxout_nodes: public nodes
{
private:
    dispatcher_group<concurrent_forward_dispatcher<edge,edge>>  fwd_dispatch_;
    dispatcher_group<concurrent_backward_dispatcher<edge,edge>> bwd_dispatch_;

    std::vector<std::unique_ptr<max_accumulator>>      fwd_accumulators_;
    std::vector<std::unique_ptr<backward_accumulator>> bwd_accumulators_;

    std::vector<cube_p<real>>    fs_      ;
    std::vector<cube_p<int>>     is_      ;
    std::vector<int>             fwd_done_;
    waiter                       waiter_  ;


public:
    maxout_nodes( size_t s,
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
        , is_(s)
        , fwd_done_(s)
        , waiter_(s)
    {
        for ( size_t i = 0; i < nodes::size(); ++i )
        {
            fwd_accumulators_[i]
                = std::make_unique<max_accumulator>();
            bwd_accumulators_[i]
                = std::make_unique<backward_accumulator>(fsize);
        }

        auto type = op.require_as<std::string>("type");
        ZI_ASSERT(type=="maxout");
    }

    options serialize() const
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

    std::vector<cube_p<int>>& get_indices_maps()
    {
        return is_;
    }

private:
    void do_forward(size_t n)
    {
        auto r = fwd_accumulators_[n]->reset();
        fs_[n] = r.first;
        is_[n] = r.second;

        //STRONG_ASSERT(!fwd_done_[n]);
        fwd_done_[n] = true;

        if ( nodes::is_output() )
        {
            waiter_.one_done();
        }
        else
        {
            fwd_dispatch_.dispatch(n,fs_[n],nodes::manager());
        }
    }

public:
    void forward(size_t n) override
    {
        forward(n, cube_p<real>(), -1);
    }

    void forward(size_t n, cube_p<real>&& f, int idx)
    {
        ZI_ASSERT(n<nodes::size());
        if ( fwd_accumulators_[n]->add(std::move(f),idx) )
        {
            do_forward(n);
        }
    }

private:
    void do_backward(size_t n, cube_p<real> const & g)
    {
        //STRONG_ASSERT(fwd_done_[n]);
        fwd_done_[n] = false;

        bwd_dispatch_.dispatch(n,g,nodes::manager());
    }

public:
    void backward(size_t n) override
    {
        backward(n, cube_p<real>());
    }

    void backward(size_t n, cube_p<real>&& g) override
    {
        ZI_ASSERT(n<nodes::size());
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
    size_t attach_maxout_edge(size_t i, edge* e)
    {
        ZI_ASSERT(i<nodes::size());
        bwd_dispatch_.sign_up(i,e);
        return fwd_accumulators_[i]->increment();
    }

    void attach_out_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<nodes::size());
        fwd_dispatch_.sign_up(i,e);
        bwd_accumulators_[i]->grow(1);
    }

    size_t attach_out_fft_edge(size_t n, edge* e) override
    {
        ZI_ASSERT(n<nodes::size());
        fwd_dispatch_.sign_up(n,nodes::fsize(),e);
        return bwd_accumulators_[n]->grow_fft(nodes::fsize(),1);
    }

    void set_eta( real /*eta*/ ) override {}
    void set_momentum( real /*mom*/ ) override {}
    void set_weight_decay( real /*wd*/ ) override {}

    void wait() override { waiter_.wait(); }

    void zap() override {}

}; // class maxout_nodes

}}} // namespace znn::v4::parallel_network
