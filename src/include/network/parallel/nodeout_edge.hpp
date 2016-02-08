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
#include "../../initializator/bernoulli_init.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class nodeout_edge: public edge
{
private:
    real            ratio_; // keeping ratio
    vec3i           insize;
    phase           phase_; // TRAIN or TEST

private:
    inline real scale() const
    {
        return ratio_;
    }

public:
    nodeout_edge( nodes * in,
                  size_t inn,
                  nodes * out,
                  size_t outn,
                  task_manager & tm,
                  real p,
                  phase phs = phase::TRAIN )
        : edge(in,inn,out,outn,tm)
        , ratio_(p)
        , phase_(phs)
    {
        insize = in->fsize();

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void setup() override
    {
        if ( phase_ == phase::TRAIN || phase_ == phase::OPTIMIZE )
        {
            bool b;
            bernoulli_init<bool>(ratio_).initialize(&b,1);
            edge::enable(b);
        }
    }

    void forward( ccube_p<real> const & f ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(size(*f)==insize);

        auto fmap = get_copy(*f);
        if ( phase_ == phase::TEST )
        {
            *fmap *= scale();
        }

        out_nodes->forward(out_num, std::move(fmap));
    }

    void backward( ccube_p<real> const & g ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(insize==size(*g));

        auto gmap = get_copy(*g);

        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num, std::move(gmap));
        }
    }

    void set_phase( phase phs ) override
    {
        phase_ = phs;
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};

}}} // namespace znn::v4::parallel_network
