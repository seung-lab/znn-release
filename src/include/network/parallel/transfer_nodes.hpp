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

#include "edge.hpp"
#include "nodes.hpp"
#include "../bias.hpp"
#include "../../utils/dispatcher.hpp"
#include "../../utils/accumulator.hpp"
#include "../../utils/waiter.hpp"
#include "../../initializator/initializators.hpp"
#include "../../transfer_function/transfer_functions.hpp"
#include "../trivial/utils.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class transfer_nodes: public nodes
{
private:
    std::vector<std::shared_ptr<bias>>  biases_     ; // transfer node
    transfer_function                   func_       ; // transfer node
    int                                 do_average_ ; //  average node

    dispatcher_group<concurrent_forward_dispatcher<edge,edge>>    fwd_dispatch_;
    dispatcher_group<concurrent_backward_dispatcher<edge,edge>>   bwd_dispatch_;

    std::vector<std::unique_ptr<forward_accumulator>>  fwd_accumulators_;
    std::vector<std::unique_ptr<backward_accumulator>> bwd_accumulators_;

    std::vector<cube_p<real>>    fs_      ;
    std::vector<cube_p<real>>    gs_      ;
    std::vector<int>             fwd_done_;
    waiter                       waiter_  ;


public:
    transfer_nodes( size_t s,
                    vec3i const & fsize,
                    options const & op,
                    task_manager & tm,
                    size_t fwd_p,
                    size_t bwd_p,
                    bool is_out )
        : nodes(s,fsize,op,tm,fwd_p,bwd_p,false,is_out)
        , biases_(s)
        , func_()
        , do_average_(false)
        , fwd_dispatch_(s)
        , bwd_dispatch_(s)
        , fwd_accumulators_(s)
        , bwd_accumulators_(s)
        , fs_(s)
        , gs_(s)
        , fwd_done_(s)
        , waiter_(s)
    {

        for ( size_t i = 0; i < nodes::size(); ++i )
        {
            fwd_accumulators_[i]
                = std::make_unique<forward_accumulator>(fsize);
            bwd_accumulators_[i]
                = std::make_unique<backward_accumulator>(fsize);
        }


        auto type = op.require_as<std::string>("type");

        if ( type == "transfer" )
        {

            func_ = get_transfer_function(op);

            // initialize biases

            real eta = op.optional_as<real>("eta", 0.0001);
            real mom = op.optional_as<real>("momentum", 0.0);
            real wd  = op.optional_as<real>("weight_decay", 0.0);

            // shared bias
            bool need_init = true;
            if ( op.contains("shared") )
            {
                auto name = op.require_as<std::string>("shared");
                if ( bias::shared_biases_pool.count(name) == 0 )
                {
                    auto& shared = bias::shared_biases_pool[name];
                    shared.resize(s);
                    for ( auto& b: shared )
                    {
                        b = std::make_shared<bias>(eta, mom, wd);
                    }
                }
                else
                {
                    need_init = false;
                }
                biases_ = bias::shared_biases_pool[name];
            }
            else
            {
                for ( auto& b: biases_ )
                {
                    b = std::make_shared<bias>(eta, mom, wd);
                }
            }

            if ( need_init )
            {
                std::string bias_values;

                if ( op.contains("biases") )
                {
                    bias_values = op.require_as<std::string>("biases");
                }
                else
                {
                    real biases_raw[nodes::size()];
                    if ( op.contains("init") )
                    {
                        auto initf = get_initializator(op);
                        initf->initialize( biases_raw, nodes::size() );
                    }
                    else
                    {
                        std::fill_n(biases_raw, nodes::size(), 0);
                    }

                    bias_values =
                        std::string( reinterpret_cast<char*>(biases_raw),
                                     sizeof(real) * nodes::size() );
                }

                load_biases(biases_, bias_values);
            }
        }
        else
        {
            if ( type == "average" )
            {
                do_average_ = true;
            }
            else
            {
                ZI_ASSERT(type=="sum");
            }
        }
    }

    virtual ~transfer_nodes() override
    {
        if ( nodes::opts().contains("shared") )
        {
            auto name = nodes::opts().require_as<std::string>("shared");
            bias::shared_biases_pool.erase(name);
        }
    }

    void set_eta( real eta ) override
    {
        if ( func_ )
        {
            nodes::opts().push("eta", eta);
            for ( auto& b: biases_ ) b->eta() = eta;
        }
    }

    void set_momentum( real mom ) override
    {
        if ( func_ )
        {
            nodes::opts().push("momentum", mom);
            for ( auto& b: biases_ ) b->momentum() = mom;
        }
    }

    void set_weight_decay( real wd ) override
    {
        if ( func_ )
        {
            nodes::opts().push("weight_decay", wd);
            for ( auto& b: biases_ ) b->weight_decay() = wd;
        }
    }

    options serialize() const override
    {
        options ret = nodes::opts();
        if ( func_ ) ret.push("biases", save_biases(biases_));
        return ret;
    }

    void debug_info() const override
    {
        nodes::debug_info();

        // mean absolute activation
        real s = 0.0;
        for ( size_t i = 0; i < nodes::size(); ++i )
            if ( enabled_[i] )
            {
                auto r = abs(*fs_[i]);
                s += mean(*r);
            }
        real val = s/nodes::num_enabled();
        std::cout << "\tmean absolute activation: " << val << "\n";

        // mean absolute gradient
        s = 0.0;
        for ( size_t i = 0; i < nodes::size(); ++i )
            if ( enabled_[i] )
            {
                auto r = abs(*gs_[i]);
                s += mean(*r);
            }
        val = s/nodes::num_enabled();
        std::cout << "\tmean absolute gradient: " << val << "\n";

        // mean absolute bias
        s = 0.0;
        for ( size_t i = 0; i < nodes::size(); ++i )
            if ( enabled_[i] )
            {
                s += std::abs(biases_[i]->b());
            }
        val = s/nodes::num_enabled();
        std::cout << "\tmean absolute bias: " << val << "\n";

        // mean absolute bias gradient
        s = 0.0;
        for ( size_t i = 0; i < nodes::size(); ++i )
            if ( enabled_[i] )
            {
                s += std::abs(sum(*gs_[i]));
            }
        val = s/nodes::num_enabled();
        std::cout << "\tmean absolute bias gradient: " << val << "\n";
    }


public:

    size_t num_out_nodes() override { return nodes::size(); }
    size_t num_in_nodes()  override { return nodes::size(); }

    std::vector<cube_p<real>>& get_featuremaps() override
    {
        for ( size_t i = 0; i < nodes::size(); ++i )
            if ( !enabled_[i] ) fs_[i] = nullptr;

        return fs_;
    }

    std::vector<cube_p<real>>& get_gradientmaps() override
    {
        for ( size_t i = 0; i < nodes::size(); ++i )
            if ( !enabled_[i] ) gs_[i] = nullptr;

        return gs_;
    }

private:
    void do_forward(size_t n)
    {
        ZI_ASSERT(enabled_[n]);

        fs_[n] = fwd_accumulators_[n]->reset();

        if ( do_average_ )
        {
            auto m = fwd_accumulators_[n]->effectively_required();
            ZI_ASSERT(m);
            *fs_[n] /= m;
        }

        //STRONG_ASSERT(!fwd_done_[n]);
        fwd_done_[n] = true;

        if ( func_ )
        {
            func_.apply(*fs_[n], biases_[n]->b());
        }

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
    void forward(size_t n, cube_p<real>&& f) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !enabled_[n] ) return;

        if ( fwd_accumulators_[n]->add(std::move(f)) )
        {
            do_forward(n);
        }
    }

    void forward(size_t n, size_t b, cube_p<complex>&& f) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !enabled_[n] ) return;

        if ( fwd_accumulators_[n]->add(b,std::move(f)) )
        {
            do_forward(n);
        }
    }


private:
    void do_backward(size_t n, cube_p<real> const & g)
    {
        ZI_ASSERT(enabled_[n]);

        gs_[n] = g;

        if ( do_average_ )
        {
            auto m = fwd_accumulators_[n]->effectively_required();
            ZI_ASSERT(m);
            *gs_[n] /= m;
        }

        //STRONG_ASSERT(fwd_done_[n]);
        fwd_done_[n] = false;

        if ( func_ )
        {
            STRONG_ASSERT(fs_[n]);
            // if ( !fs_[n] )
            // {
            //     std::cout << "N: " << n <<
            //         ( nodes::is_output() ? " output\n" : "no\n");
            //     STRONG_ASSERT(0);
            // }
            func_.apply_grad(*gs_[n],*fs_[n]);
            biases_[n]->update(sum(*gs_[n]),patch_sz_);
            // fs_[n].reset();
        }
        bwd_dispatch_.dispatch(n,gs_[n],nodes::manager());
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

    void backward(size_t n, size_t b, cube_p<complex>&& g) override
    {
        ZI_ASSERT((n<nodes::size())&&(!nodes::is_output()));
        if ( !enabled_[n] ) return;

        if ( bwd_accumulators_[n]->add(b,std::move(g)) )
        {
            do_backward(n,bwd_accumulators_[n]->reset());
        }
    }

    void attach_out_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<nodes::size());
        fwd_dispatch_.sign_up(i,e);
        bwd_accumulators_[i]->grow(1);
    }

    void attach_in_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<nodes::size());
        bwd_dispatch_.sign_up(i,e);
        fwd_accumulators_[i]->grow(1);
    }

    size_t attach_out_fft_edge(size_t n, edge* e, vec3i const & s) override
    {
        ZI_ASSERT(n<nodes::size());
        fwd_dispatch_.sign_up(n,s,e);
        return bwd_accumulators_[n]->grow_fft(s,1);
    }

    size_t attach_in_fft_edge(size_t n, edge* e, vec3i const & s) override
    {
        ZI_ASSERT(n<nodes::size());
        bwd_dispatch_.sign_up(n,s,e);
        return fwd_accumulators_[n]->grow_fft(s,1);
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
        gs_[n].reset();

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
        fwd_accumulators_[n]->enable_all(false);

        // reset feature map
        fs_[n].reset();
        gs_[n].reset();

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
        bwd_dispatch_.enable(n,b);

        fwd_accumulators_[n]->enable_all(b);
        bwd_accumulators_[n]->enable_all(b);

        // reset feature map
        fs_[n].reset();
        gs_[n].reset();

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

    void enable_in_edge(size_t n, bool b) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !fwd_accumulators_[n]->enable(b) )
            disable_fwd(n);
    }

    void enable_out_fft_edge(size_t n, bool b, vec3i const & s) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !bwd_accumulators_[n]->enable_fft(s,b) )
            disable_bwd(n);
    }

    void enable_in_fft_edge(size_t n, bool b, vec3i const & s) override
    {
        ZI_ASSERT(n<nodes::size());
        if ( !fwd_accumulators_[n]->enable_fft(s,b) )
            disable_fwd(n);
    }

    void wait() override { waiter_.wait(); }

    void zap() override {}

}; // class transfer_nodes

}}} // namespace znn::v4::parallel_network
