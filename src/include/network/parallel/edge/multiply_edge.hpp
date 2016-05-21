//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
//                    2016  Kisuk Lee           <kisuklee@mit.edu>
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

class multiply_edge: public edge
{
private:
    real    epsilon_; // small constant for numerical stability

private:
    cube_p<real> do_backward( ccube<real> const & g )
    {
        auto grad = get_copy(g);
        auto fmap = in_nodes->get_featuremaps()[in_num];

        size_t n = grad->num_elements();
        for ( size_t i = 0; i < n; ++i )
        {
            grad->data()[i] /= (fmap->data()[i] + epsilon_);
        }

        // DEBUG
        // std::cout << "[" << edge::name() << "] " << "\n";
        // std::cout << *grad << "\n\n";

        return grad;
    }

public:
    multiply_edge( nodes * in,
                   size_t inn,
                   nodes * out,
                   size_t outn,
                   task_manager & tm,
                   real eps )
        : edge(in,inn,out,outn,tm)
        , epsilon_(eps)
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        if ( !enabled_ ) return;

        out_nodes->forward(out_num, get_copy(*f));
    }

    void backward( ccube_p<real> const & g ) override
    {
        if ( !enabled_ ) return;

        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num, do_backward(*g));
        }
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};

}}} // namespace znn::v4::parallel_network
