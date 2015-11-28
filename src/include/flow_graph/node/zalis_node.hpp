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
#include "computation/constrain_affinity.hpp"
#include "computation/zalis.hpp"

namespace znn { namespace v4 { namespace flow_graph {

class zalis_node: public unidirectional_node
{
private:
    bool constrained_ = false;
    bool frac_norm_   = false;
    real high_        = 1;
    real low_         = 0;

private:
    void setup( options const & op )
    {
        // [TODO]
        // params setup

        // [TODO]
        // sanity check
    }

    interface_type compute( interface_type && in )
    {
        std::pair<tensor, tensor> inputs;
        for ( auto& i: in )

        zalis_weight weight;

        if ( constrained_ )
        {
            // merger constrained
            auto T = constrain_affinity(true_affs, affs, zalis_phase::MERGER);
            auto m = zalis(T, affs, frac_norm, 1, low);
            weight.merger = m.merger;

            // splitter constrained
            T = constrain_affinity(true_affs, affs, zalis_phase::SPLITTER);
            auto s = zalis(T, affs, frac_norm, high, 0);
            weight.splitter = s.splitter;
        }
        else
        {
            weight = zalis(true_affs, affs, frac_norm, high, low);
        }

        fwd_load(name() + ":merger", weight.merger);
        fwd_load(name() + ":splitter", weight.splitter);
    }

protected:
    void forward() override
    {
        fwd_load(compute(bottoms()));
        fwd_dispatch();
    }

    void backward() override
    {
        UNIMPLEMENTED();
    }

public:
    void setup() override
    {
        setup(opts());
    }

public:
    interface_type forward( interface_type && in ) override
    {
        return compute(std::move(in));
    }

    interface_type backward( interface_type && out ) override
    {
        UNIMPLEMENTED();
    }

public:
    explicit zalis_node( options const & op )
        : unidirectional_node(op)
    {}

    virtual ~zalis_node() {}

}; // class zalis_node

}}} // namespace znn::v4::flow_graph