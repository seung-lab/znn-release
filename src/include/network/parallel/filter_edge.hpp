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
#include "edges_fwd.hpp"
#include "nodes.hpp"

#include "../../convolution/convolution.hpp"
#include "../filter.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class filter_edge: public edge
{
private:
    vec3i    filter_stride;
    filter & filter_;

    ccube_p<real> last_input;

    task_manager::task_handle pending_ = 0;

private:
    void do_forward( ccube_p<real> const & f )
    {
        ZI_ASSERT(enabled_);

        last_input = f;

        out_nodes->forward(out_num,
            convolve_sparse(*f, filter_.W(), filter_stride));
    }

    void do_update( ccube_p<real> const & g )
    {
        ZI_ASSERT(enabled_);

        auto dEdW = convolve_sparse_flipped(*last_input, *g, filter_stride);
        filter_.update(*dEdW, patch_sz_);
    }

public:
    filter_edge( nodes * in,
                 size_t inn,
                 nodes * out,
                 size_t outn,
                 task_manager & tm,
                 vec3i const & stride,
                 filter & f )
        : edge(in,inn,out,outn,tm), filter_stride(stride), filter_(f)
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        if ( !enabled_ ) return;

        manager.require_done( pending_, &filter_edge::do_forward, this, f );
    }

    void backward( ccube_p<real> const & g ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(last_input);
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num,
                convolve_sparse_inverse(*g, filter_.W(), filter_stride));
        }

        pending_ = manager.schedule_unprivileged(&filter_edge::do_update,
                                                 this, g);
    }

    void zap(edges* e) override
    {
        // guard gg(m);
        manager.require_done(pending_,&edges::edge_zapped,e);
        //e->edge_zapped();
    }
};


}}} // namespace znn::v4::parallel_network
