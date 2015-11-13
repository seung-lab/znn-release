//
// Copyright (C)      2015  Kisuk Lee           <kisuklee@mit.edu>
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

#include "node.hpp"
#include "../../network/parallel/network.hpp"

namespace znn { namespace v4 { namespace flow_graph {

using namespace parallel_network;

class network_node: public bidirectional_node
{
private:
    std::unique_ptr<network> net_;

private:
    void construct_network( options const & op )
    {
        // [TODO]
        // Should we replace the current net with a new one,
        // or just keep the current one?
        if ( net_ ) return;

        // network spec
        std::vector<options> nodes;
        std::vector<options> edges;
        auto net_spec = op.required_as<std::string>("net_spec");
        parse_net_file(nodes, edges, net_spec);

        // [TODO]
        // Ideally, output size of the network at this node should be
        // determined dynamically, based on the output size of the entire
        // flow graph.
        //
        // Currently, we just set it by taking user input.

        // output size
        auto out_sz = op.require_as<ovec3i>("size");

        // thread count
        auto tc = op.optional_as<size_t>("tc",0);
        if ( !tc ) tc = std::thread::hardware_concurrency();

        // force fft
        auto force_fft = op.optional_as<bool>("fft",false);
        if ( force_fft ) network::force_fft(edges)

        // optimization
        // 0 indicates no optimization,
        // otherwise the number of interations for optimization
        auto optimize = op.optional_as<size_t>("optimize",0);
        if ( optimize ) network::optimize(nodes, edges, out_sz, tc, optimize);

        // [TODO]
        // Currently only support "TRAIN" network.

        // construct network
        net_ = std::make_unique<network>(nodes, edges, out_sz, tc);

        // [TODO]
        // hyperparameter setup
    }

protected:
    void forward() override
    {
        auto prop = net_->forward(bottoms());

        fwd_load(prop);
        fwd_dispatch();
    }

    void backward() override
    {
        auto grad = net_->backward(tops());

        // [TODO]
        // Currently grad from network is empty.
        // This should be modified later to allow recurrent nets, i.e.,
        // back propagation through time (BPTT).
        bwd_load(grad);
        bwd_dispatch();
    }

public:
    void setup() override
    {
        construct_network(opts());
    }

public:
    interface_type forward( interface_type && in ) override
    {
        return net_->forward(std::move(in));
    }

    interface_type backward( interface_type && out ) override
    {
        return net_->backward(std::move(out));
    }

public:
    explicit network_node( options const & op )
        : bidirectional_node(op)
    {}

    virtual ~network_node() {}

}; // class network_node

}}} // namespace znn::v4::flow_graph