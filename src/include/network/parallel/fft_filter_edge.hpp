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

#include "../../fft/fftw.hpp"
#include "../filter.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class fft_filter_edge: public edge
{
private:
    vec3i    filter_stride;
    filter & filter_;

#ifndef ZNN_DONT_CACHE_FFTS
    ccube_p<complex> w_fft;
#endif
    ccube_p<complex> last_input;

    size_t fwd_bucket_;
    size_t bwd_bucket_;

    task_manager::task_handle pending_ = 0;

    fftw::transformer fftw_;

private:
    void do_forward( ccube_p<complex> const & f )
    {
        ZI_ASSERT(enabled_);

        last_input = f;

#ifdef ZNN_DONT_CACHE_FFTS
        auto w_fft = get_w_fft();
#endif
        auto fw = *w_fft * *f;
        out_nodes->forward(out_num, fwd_bucket_, std::move(fw));
        //out_nodes->forward(out_num, fwd_bucket_, w_fft, f);
    }

    void do_update( ccube_p<complex> const & g )
    {
        ZI_ASSERT(enabled_);

        auto dEdW_fft = *last_input * *g;
        auto dEdW = fftw_.backward(std::move(dEdW_fft));
        real norm = dEdW->num_elements();

        if ( fftw_.size() != fftw_.actual_size() )
        {
            dEdW = crop_left(*dEdW, fftw_.size());
        }

        flip(*dEdW);
        // TODO(zlateski): WTH was happening with sparse_implode before
        //                 when I had to use sparse_implode_slow
        //                 ony happened on my laptop
        dEdW = sparse_implode_slow(*dEdW, filter_stride, size(filter_.W()));
        *dEdW /= norm;

        filter_.update(*dEdW, patch_sz_);

#ifndef ZNN_DONT_CACHE_FFTS
        initialize();
#endif
    }

#ifndef ZNN_DONT_CACHE_FFTS
    void initialize()
    {
        w_fft = get_w_fft();
    }
#endif

    cube_p<complex> get_w_fft()
    {
        // TODO(zlateski): WTH was happening with sparse_exploce before
        //                 when I had to use sparse_explode_slow,
        //                 ony happened on my laptop
        auto w_tmp = sparse_explode_slow(filter_.W(), filter_stride,
                                         fftw_.actual_size());
//                                         in_nodes->fsize());
        return fftw_.forward(std::move(w_tmp));
    }


public:
    fft_filter_edge( nodes * in,
                     size_t inn,
                     nodes * out,
                     size_t outn,
                     task_manager & tm,
                     vec3i const & stride,
                     filter & f )
        : edge(in,inn,out,outn,tm), filter_stride(stride), filter_(f)
        , fftw_(in_nodes->fsize())
    {
        bwd_bucket_ = in->attach_out_fft_edge(inn, this);
        fwd_bucket_ = out->attach_in_fft_edge(outn, this, in->fsize());
#ifndef ZNN_DONT_CACHE_FFTS
        pending_ = manager.schedule_unprivileged(&fft_filter_edge::initialize,this);
#endif
    }

    void forward( ccube_p<complex> const & f ) override
    {
        if ( !enabled_ ) return;

        manager.require_done( pending_, &fft_filter_edge::do_forward, this, f );
    }

    void backward( ccube_p<complex> const & g ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(last_input);

        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, bwd_bucket_, cube_p<complex>());
        }
        else
        {
#ifdef ZNN_DONT_CACHE_FFTS
            auto w_fft = get_w_fft();
#endif
            auto grad = *w_fft * *g;
            in_nodes->backward(in_num, bwd_bucket_, std::move(grad));
        }

        pending_ = manager.schedule_unprivileged(
                                &fft_filter_edge::do_update, this, g);
    }

    void zap(edges* e) override
    {
        manager.require_done(pending_,&edges::edge_zapped,e);
    }
};

}}} // namespace znn::v4::parallel_network
