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

class normalize_edge: public edge
{
private:
    bool    global_stat_     ;
    real    moving_avg_frac_ ;
    real    epsilon_         ;
    phase   phase_           ;

    cube_p<real> normalized_ ;

    // for moving average calculation
    real    moving_win_ = 0.0;
    real    moving_avg_ = 0.0;
    real    moving_var_ = 0.0;

private:
    inline real scale() const
    {
        return moving_win_ == static_cast<real>(0)
             ? static_cast<real>(0)
             : static_cast<real>(1)/moving_win_;
    }

    cube_p<real> normalize( ccube_p<real> const & f )
    {
        auto r = get_copy(*f);

        real avg;
        real var;

        // use the stored mean/variance estimates
        if ( global_stat_ )
        {
            avg = scale()*moving_avg_;
            var = scale()*moving_var_;
        }
        else // local estimates
        {
            avg = mean(*f);
            var = variance(*f);
        }

        size_t m = f->num_elements();
        for ( size_t i = 0; i < m; ++i )
        {
            r->data()[i] -= avg;
            r->data()[i] /= (var + epsilon_);
        }

        // for later use
        normalized_ = get_copy(r);

        // moving window
        moving_win_ *= moving_avg_frac_;
        moving_win_ += static_cast<real>(1);

        // moving average
        moving_avg_ *= moving_avg_frac_;
        moving_avg_ += avg;

        // moving variance
        real c = m > 1 ? static_cast<real>(m)/(m - 1)
                       : static_cast<real>(1);
        moving_var_ *= moving_avg_frac_;
        mvoing_var_ += c*var;

        return r;
    }

public:
    normalize_edge( nodes * in,
                    size_t inn,
                    nodes * out,
                    size_t outn,
                    task_manager & tm,
                    bool stat,
                    real frac,
                    real eps,
                    phase phs = phase::TRAIN )
        : edge(in,inn,out,outn,tm)
        , global_stat_(stat)
        , moving_avg_frac_(frac)
        , epsilon_(eps)
        , phase_(phs)
        , normalized_()
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void setup() override
    {
        if ( phase_ == phase::TEST )
        {
            global_stat_ = true;
        }
    }

    void forward( ccube_p<real> const & f ) override
    {
        if ( !enabled_ ) return;

        out_nodes->forward(out_num, normalize(f));
    }

    void backward( ccube_p<real> const & g ) override
    {
        if ( !enabled_ ) return;
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
