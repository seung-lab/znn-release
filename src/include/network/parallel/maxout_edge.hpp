//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
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

#include "../../utils/max_accumulator.hpp"
#include "edges_fwd.hpp"
#include "maxout_nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class maxout_edge: public edge
{
private:
    maxout_nodes * target;
    int group_idx;

private:
    cube_p<real> maxout_backprop(ccube<real> const & g)
    {
        auto         r = get_cube<real>(size(g));
        real*       rp = r->data();
        const real* gp = g.data();
        const int*  ip = target->get_indices_maps()[out_num]->data();

        for ( size_t i = 0; i < r->num_elements(); ++i )
        {
            if ( ip[i] == group_idx )
                rp[i] = gp[i];
            else
                rp[i] = 0;
        }

        return r;
    }

public:
    maxout_edge( nodes * in,
                 size_t inn,
                 nodes * out,
                 size_t outn,
                 task_manager & tm )
        : edge(in,inn,out,outn,tm)
    {
        ZI_ASSERT(inn=outn);
        in->attach_out_edge(inn,this);

        // TODO: any better solution?
        target = reinterpret_cast<maxout_nodes*>(out_nodes);
        auto idx = target->attach_maxout_edge(outn,this);
        group_idx = static_cast<int>(idx);
    }

    void forward( ccube_p<real> const & f ) override
    {
        target->forward(out_num, get_copy(*f), group_idx);
    }

    void backward( ccube_p<real> const & g ) override
    {
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num, maxout_backprop(*g));
        }
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};

}}} // namespace znn::v4::parallel_network
