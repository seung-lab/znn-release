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

#include "../../assert.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"
#include "../../utils/task_manager.hpp"
#include "nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class edges;

class edge
{
protected:
    nodes * in_nodes ;
    size_t  in_num   ;
    nodes * out_nodes;
    size_t  out_num  ;

    task_manager & manager;

    size_t fwd_priority_;
    size_t bwd_priority_;

    // enable/disable
    bool   enabled_ = true;

protected:
    static phase phase_;

public:
    static void set_phase( phase phs ) { phase_ = phs; }

public:
    edge( nodes * in, size_t inn, nodes * out, size_t outn, task_manager & m )
        : in_nodes(in), in_num(inn), out_nodes(out), out_num(outn), manager(m)
    {
        fwd_priority_ = out->fwd_priority() * 1024 + outn;
        bwd_priority_ = in->bwd_priority() * 1024 + inn;
    }

    size_t fwd_priority() const { return fwd_priority_; }
    size_t bwd_priority() const { return bwd_priority_; }

    bool is_enabled() const { return enabled_; }

    std::string name() const
    {
        return in_nodes->name() + ":" + std::to_string(in_num) + "_" +
               out_nodes->name() + ":" + std::to_string(out_num);
    }

    virtual ~edge() {}

    virtual void setup() {}

    virtual void forward( ctensor<real> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ctensor<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ctensor<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ctensor<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void enable(bool b)
    {
        if ( enabled_ == b ) return;

        enabled_ = b;
        in_nodes->enable_out_edge(in_num,b);
        out_nodes->enable_in_edge(out_num,b);
    }

    virtual void enable_fwd(bool b)
    {
        if ( enabled_ == b ) return;

        enabled_ = b;
        out_nodes->enable_in_edge(out_num,b);
    }

    virtual void enable_bwd(bool b)
    {
        if ( enabled_ == b ) return;

        enabled_ = b;
        in_nodes->enable_out_edge(in_num,b);
    }

    virtual void zap(edges*) = 0;
};

phase edge::phase_ = phase::TRAIN;

}}} // namespace znn::v4::parallel_network
