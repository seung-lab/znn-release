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
#include "../../initializator/bernoulli_init.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class dropout_edge: public edge
{
private:
    real            ratio_; // keeping ratio
    cube_p<bool>    mask_ ; // dropout mask
    bool            force_; // dropout regardless of phase
    vec3i           insize;

private:
    inline real scale() const
    {
        return static_cast<real>(1)/ratio_;
    }

    // inplace dropout
    inline void dropout( cube<real> & c, ccube<bool> const & msk ) const
    {
        size_t s = c.num_elements();
        for ( size_t i = 0; i < s; ++i )
        {
            if ( msk.data()[i] )
                c.data()[i] *= scale();
            else
                c.data()[i] = static_cast<real>(0);
        }
    }

    inline tensor<real> dropout_forward( tensor<real> const & f )
    {
        tensor<real> ret = get_copy(f);
        if ( force_ || edge::phase_ != phase::TEST )
        {
            if ( !mask_ ) mask_ = get_cube<bool>(size(f));
            bernoulli_init<bool>(ratio_).initialize(mask_); // new random mask
            for ( auto& c: ret ) dropout(*c,*mask_);
        }
        return ret;
    }

    inline tensor<real> dropout_backward( tensor<real> const & g )
    {
        tensor<real> ret = get_copy(g);
        if ( force_ || edge::phase_ != phase::TEST )
        {
            ZI_ASSERT(mask_);
            for ( auto& c: ret ) dropout(*c,*mask_);
            mask_.reset();
        }
        return ret;
    }

public:
    dropout_edge( nodes * in,
                  size_t inn,
                  nodes * out,
                  size_t outn,
                  task_manager & tm,
                  real p,
                  bool force = false )
        : edge(in,inn,out,outn,tm)
        , ratio_(p)
        , mask_()
        , force_(force)
        , insize(in->fsize())
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ctensor<real> const & f ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(size(f)==insize);
        out_nodes->forward(out_num, dropout_forward(f));
    }

    void backward( ctensor<real> const & g ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(size(g)==insize);
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            in_nodes->backward(in_num, dropout_backward(g));
        }
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};

}}} // namespace znn::v4::parallel_network
