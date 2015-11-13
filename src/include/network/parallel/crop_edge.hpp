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

#include "edges_fwd.hpp"
#include "nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class crop_edge: public edge
{
private:
    vec3i   offset;
    vec3i   insize;

private:
    inline vec3i crop_size() const
    {
        return insize - offset - offset;
    }

public:
    crop_edge( nodes * in,
               size_t inn,
               nodes * out,
               size_t outn,
               task_manager & tm,
               vec3i const & off )
        : edge(in,inn,out,outn,tm)
        , offset(off)
    {
        insize = in->fsize();

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(size(*f)==insize);
        out_nodes->forward(out_num, crop(*f,offset,crop_size()));
    }

    void backward( ccube_p<real> const & g ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(insize==size(*g));

        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            auto gmap = get_cube<real>(insize);
            in_nodes->backward(in_num,
                               pad_zeros(*g,offset,pad_style::BOTH));
        }
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};

}}} // namespace znn::v4::parallel_network
