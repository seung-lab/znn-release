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
#include "../../utils/waiter.hpp"

namespace znn { namespace v4 { namespace flow_graph {

class interface_node
{
private:
    waiter  waiter_;

protected:
    void forward( interface_type && in )
    {
        waiter_.one_done();
    }

    void backward( interface_type && in )
    {
        waiter_.one_done();
    }

public:
    bool is_bidirectional() const override
    {
        return true;
    }

public:
    interface_type & forward( interface_type && in )
    {
        node::fwd_load(std::move(in));
        node::fwd_dispatch();

        waiter_.wait();
        return get_bottoms();
    }

    interface_type & backward( interface_type && out )
    {
        node::bwd_load(std::move(out));
        node::bwd_dispatch();

        waiter_.wait();
        return get_tops();
    }

public:
    interface_node( options const & op )
        : node(op)
        , waiter_(1)
    {}

    virtual ~interface_node() {}
};

}}} // namespace znn::v4