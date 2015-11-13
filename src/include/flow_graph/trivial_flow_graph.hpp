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

#include "node/nodes.hpp"

#include <boost/algorithm/string.hpp>

namespace znn { namespace v4 {

class trivial_flow_graph
{
private:
    struct graph_node
    {
        options const * opts;

        std::vector<std::string> tops;
        std::vector<std::string> btms;

        std::unique_ptr<flow_graph::node> dnode;
    };

private:
    std::map<std::string, graph_node*>  nodes_;

    graph_node* interface_node_;

public:
    ~trivial_flow_graph()
    {
        for ( auto& n: nodes_ ) delete n.second;
    }

private:
    void add_node( options const & op )
    {
        auto name = op.require_as<std::string>("name");
        auto type = op.require_as<std::string>("type");

        ZI_ASSERT(nodes_.count(name)==0);
        graph_node* n = new graph_node;
        n->opts = &op;
        nodes_[name] = n;

        if ( type == "interface" )
        {
            ZI_ASSERT(!interface_node_);
            interface_node_ = n;
        }
    }

    void create_nodes()
    {
        for ( auto& n: nodes_ )
        {
            auto type = n.second->opts->require_as<std::string>("type");

            if ( type == "interface" )
            {
                UNIMPLEMENTED();
                // n.second->dnode =
                //     std::make_unique<flow_graph::interface_node>
                //         (blah blah blah);
            }
            else if ( type == "network" )
            {
                UNIMPLEMENTED();
                // n.second->dnode =
                //     std::make_unique<flow_graph::network_node>
                //         (blah blah blah);
            }
            else if ( type == "ZALIS" )
            {
                UNIMPLEMENTED();
                // n.second->dnode =
                //     std::make_unique<flow_graph::zalis_node>
                //         (blah blah blah);
            }
            else if ( type == "KALIS" )
            {
                UNIMPLEMENTED();
                // n.second->dnode =
                //     std::make_unique<flow_graph::kalis_node>
                //         (blah blah blah);
            }
            else
            {
                throw std::logic_error(HERE() + "unknown node type: " + type);
            }

            n.second->opts = nullptr;
        }
    }

    void create_flows()
    {
        for ( auto& n: nodes_ )
        {
            auto name = n.first;
            auto opts = n.second->opts;
            auto dest = n.second->dnode;

            // bottoms
            auto btms = opts.require_as<ovector<std::string>>("bottoms");
            for ( auto& btm: btms )
            {
                std::vector<std::string> parts;
                boost::split(parts, btm, boost::is_any_of(":"));
                ZI_ASSERT(parts.size()==2);

                // forward flow
                auto src = nodes_[parts[0]]->d;
                src->add_target(parts[1], dest);

                // backward flow
                if ( src->is_bidirectional() )
                    dest->add_target(parts[1], src);
            }
        }
    }

public:
    trivial_flow_graph( std::string const & spec )
        : nodes_()
    {
        std::vector<options> ns, es;
        parse_net_file(ns, es, spec);

        for ( auto& n: ns ) add_node(n);

        create_nodes();
        create_flows();
    }

    std::map<std::string, std::pair<vec3i,size_t>> inputs() const
    {
        std::map<std::string, std::pair<vec3i,size_t>> ret;
        // TOOD
        return ret;
    }

    std::map<std::string, std::pair<vec3i,size_t>> outputs() const
    {
        std::map<std::string, std::pair<vec3i,size_t>> ret;
        // TODO
        return ret;
    }

    void setup()
    {
        for ( auto& n: ns ) n->dnode->setup();
    }

    flow_graph::interface_type
    forward( flow_graph::interface_type && in )
    {
        auto top =
            reinterpret_cast<interface_node*>
                (interface_node_->dnode->get());

        return top->forward(std::move(in));
    }

    flow_graph::interface_type
    backward( flow_graph::interface_type && out )
    {
        auto bottom =
            reinterpret_cast<interface_node*>
                (interface_node_->dnode->get());

        return bottom->backward(std::move(out));
    }

};

}} // namespace znn::v4