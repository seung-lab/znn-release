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

namespace znn { namespace v4 { namespace flow_graph {

class interface_node: public bidirectional_node
{
protected:
    void forward()  override {}
    void backward() override {}

public:
    void setup() override {}

public:
    interface_type forward( interface_type && in ) override
    {
        fwd_load(std::move(in));
        fwd_dispatch();

        return bottoms();
    }

    interface_type backward( interface_type && out ) override
    {
        bwd_load(std::move(out));
        bwd_dispatch();

        return tops();
    }

public:
    explicit interface_node( options const & op )
        : bidirectional_node(op)
    {}

    virtual ~interface_node() {}
};

}}} // namespace znn::v4::flow_graph