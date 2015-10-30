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

    // minibatch averaging
    real   patch_sz_ = 1;

    // on/off
    bool   enabled_ = true;

public:
    edge( nodes * in, size_t inn, nodes * out, size_t outn, task_manager & m )
        : in_nodes(in), in_num(inn), out_nodes(out), out_num(outn), manager(m)
    {
        fwd_priority_ = out->fwd_priority() * 1024 + outn;
        bwd_priority_ = in->bwd_priority() * 1024 + inn;
    }

    size_t fwd_priority() const { return fwd_priority_; }
    size_t bwd_priority() const { return bwd_priority_; }

    void set_patch_size( real s )
    {
        ZI_ASSERT(s > 0);
        patch_sz_ = s;
    }

    std::string name() const
    {
        return in_nodes->name() + ":" + std::to_string(in_num) + "_" +
               out_nodes->name() + ":" + std::to_string(out_num);
    }

    virtual ~edge() {}

    virtual void forward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void enable_fwd(bool b)
    {
        if ( enabled_ == b ) return;

        enabled_ = b;
        if ( enabled_ )
        {
            out_nodes->enable(out_num,true);
        }
        else // disable
        {
            out_nodes->disable_in_edge(out_num);
        }
    }

    virtual void enable_bwd(bool b)
    {
        if ( enabled_ == b ) return;

        enabled_ = b;
        if ( enabled_ )
        {
            in_nodes->enable(in_num,true);
        }
        else // disabled_
        {
            in_nodes->disable_out_edge(in_num);
        }
    }

    // [kisuklee]
    // This is only temporary implementation and will be removed.
    virtual void set_phase( phase ){}

    virtual void zap(edges*) = 0;
};



}}} // namespace znn::v4::parallel_network
