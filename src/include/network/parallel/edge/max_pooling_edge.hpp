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

#include "edges_fwd.hpp"
#include "nodes.hpp"
#include "../../pooling/pooling.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class max_pooling_edge: public edge
{
private:
    vec3i filter_size  ;
    vec3i filter_stride;

    tensor<int> indices;
    vec3i       insize ;

public:
    max_pooling_edge( nodes * in,
                      size_t inn,
                      nodes * out,
                      size_t outn,
                      task_manager & tm,
                      vec3i const & size,
                      vec3i const & stride )
        : edge(in,inn,out,outn,tm)
        , filter_size(size)
        , filter_stride(stride)
        , insize(in->fsize())
    {
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ctensor<real> const & f ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(size(f)==insize);
        tensor<real> fmaps;
        indices.clear();
        for ( auto& c: f )
        {
            auto r = pooling_filter(get_copy(*c),
                                    [](real a, real b){ return a>b; },
                                    filter_size,
                                    filter_stride);

            fmaps.push_back(r.first);
            indices.push_back(r.second);
        }
        out_nodes->forward(out_num, std::move(fmaps));
    }

    void backward( ctensor<real> const & g ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(!indices.empty());
        ZI_ASSERT(indices.size()==g.size());
        ZI_ASSERT(insize==size(g)+(filter_size-vec3i::one)*filter_stride);
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            tensor<real> gmaps;
            for ( size_t i = 0; i < g.size(); ++i )
            {
                gmaps.push_back(pooling_backprop(insize, *g[i], *indices[i]));
            }
            in_nodes->backward(in_num, std::move(gmaps));
        }
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};


class real_pooling_edge: public edge
{
private:
    vec3i   filter_size;

    cube_p<int> indices;
    vec3i       insize ;

    vec3i       outsize;

public:
    real_pooling_edge( nodes * in,
                       size_t inn,
                       nodes * out,
                       size_t outn,
                       task_manager & tm,
                       vec3i const & size )
        : edge(in,inn,out,outn,tm)
        , filter_size(size)
        , insize(in->fsize())
        , outsize(in->fsize()/size)
    {
        ZI_ASSERT((insize%size)==vec3i::zero);

        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ctensor<real> const & f ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(size(f)==insize);
        tensor<real> fmaps;
        indices.clear();
        for ( auto& c: f )
        {
            auto r = pooling_filter(get_copy(*c),
                                    [](real a, real b){ return a>b; },
                                    filter_size,
                                    vec3i::one);

            fmaps.push_back(
                sparse_implode_slow(*r.first,filter_size,outsize));
            indices.push_back(
                sparse_implode_slow(*r.second,filter_size,outsize));
        }
        out_nodes->forward(out_num, std::move(fmaps));
    }

    void backward( ctensor<real> const & g ) override
    {
        if ( !enabled_ ) return;

        ZI_ASSERT(!indices.empty());
        ZI_ASSERT(indices.size()==g.size());
        ZI_ASSERT(insize==size(g)*filter_size);
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            tensor<real> gmaps;
            for ( size_t i = 0; i < g.size(); ++i )
            {
                gmaps.push_back(pooling_backprop(insize, *g[i], *indices[i]));
            }
            in_nodes->backward(in_num, std::move(gmaps));
        }
    }

    void zap(edges* e) override
    {
        e->edge_zapped();
    }
};


}}} // namespace znn::v4::parallel_network
