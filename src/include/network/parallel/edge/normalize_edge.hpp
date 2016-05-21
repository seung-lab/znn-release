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
    enum class mode : std::uint8_t {NONE = 0, LOCAL = 1, GLOBAL = 2};

private:
    real    moving_avg_frac_;
    real    epsilon_        ;
    phase   phase_          ;

    // for later use
    cube_p<real> normalized_;

    // for moving average calculation
    real &  moving_win_;
    real &  moving_avg_;
    real &  moving_var_;

    // used in both forward/backward passes
    real    avg_ = 0.0;
    real    var_ = 0.0;

    // force to use either local/global statistics regardless of phase
    mode    force_ = mode::NONE;

private:
    inline real scale() const
    {
        return moving_win_ == static_cast<real>(0)
             ? static_cast<real>(0)
             : static_cast<real>(1)/moving_win_;
    }

    inline bool use_global_stat() const
    {
        bool ret = false;

        if ( force_ == mode::NONE )
        {
            ret = (phase_ == phase::TEST);
        }
        else
        {
            ret = (force_ == mode::GLOBAL);
        }

        return ret;
    }

    cube_p<real> normalize( ccube<real> const & f )
    {
        auto r = get_copy(f);

        // use the stored mean/variance estimates
        if ( use_global_stat() )
        {
            avg_ = scale()*moving_avg_;
            var_ = scale()*moving_var_;
        }
        else // local estimates
        {
            avg_ = mean(f);
            var_ = variance(f);
        }

        size_t m = f.num_elements();
        for ( size_t i = 0; i < m; ++i )
        {
            r->data()[i] -= avg_;
            r->data()[i] /= std::sqrt(var_ + epsilon_);
        }

        // keep for backward pass
        normalized_ = get_copy(*r);

        // compute and save moving average
        if ( !use_global_stat() && phase_ != phase::TEST )
        {
            // moving window
            moving_win_ *= moving_avg_frac_;
            moving_win_ += static_cast<real>(1);

            // moving average
            moving_avg_ *= moving_avg_frac_;
            moving_avg_ += avg_;

            // moving variance
            real c = m > 1 ? static_cast<real>(m)/(m - 1)
                           : static_cast<real>(1);
            moving_var_ *= moving_avg_frac_;
            moving_var_ += c*var_;
        }

        return r;
    }

    cube_p<real> do_backward( ccube<real> const & g )
    {
        cube_p<real> r;

        if ( use_global_stat() || phase_ == phase::TEST )
        {
            r = get_copy(g);
        }
        else
        {
            // r = dE/dY - mean(dE/dY * Y) * Y
            cube<real> & y = *normalized_;
            y *= mean(*(g*y));
            r = g - y;

            // r = dE/dY - mean(dE/dY) - mean(dE/dY * Y) * Y
            *r -= mean(g);

            // normalize
            *r /= std::sqrt(var_ + epsilon_);

            normalized_.reset();
        }

        return r;
    }

public:
    normalize_edge( nodes * in,
                    size_t inn,
                    nodes * out,
                    size_t outn,
                    task_manager & tm,
                    std::string force,
                    real frac,
                    real eps,
                    filter & f,
                    phase phs = phase::TRAIN )
        : edge(in,inn,out,outn,tm)
        , moving_avg_frac_(frac)
        , epsilon_(eps)
        , moving_win_(f.W().data()[0])
        , moving_avg_(f.W().data()[1])
        , moving_var_(f.W().data()[2])
        , phase_(phs)
        , normalized_()
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);

        if ( force == "local" )
        {
            force_ = mode::LOCAL;
        }
        else if ( force == "global" )
        {
            force_ = mode::GLOBAL;
        }
        else
        {
            force_ = mode::NONE;
        }
    }

    void forward( ccube_p<real> const & f ) override
    {
        if ( !enabled_ ) return;

        out_nodes->forward(out_num, normalize(*f));
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
