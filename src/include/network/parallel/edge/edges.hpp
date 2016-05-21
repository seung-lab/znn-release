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
#include "softmax_edge.hpp"
#include "filter_edge.hpp"
#include "fft_filter_edge.hpp"
#include "filter_ds_edge.hpp"
#include "fft_filter_ds_edge.hpp"
#include "dummy_edge.hpp"
#include "max_pooling_edge.hpp"
#include "dropout_edge.hpp"
#include "nodeout_edge.hpp"
#include "crop_edge.hpp"
#include "maxout_edge.hpp"
#include "multiply_edge.hpp"
#include "normalize_edge.hpp"
#include "L2_norm_edge.hpp"
#include "nodes.hpp"
#include "../../utils/waiter.hpp"
#include "../../options/options.hpp"
#include "../filter.hpp"


namespace znn { namespace v4 { namespace parallel_network {

// convolution & deconvolution
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     vec3i const & stride,
                     task_manager & tm,
                     bool deconv,
                     filter_tag )
    : options_(opts)
    , tm_(tm)
{
    size_t n = in->num_out_nodes();
    size_t m = out->num_in_nodes();

    ZI_ASSERT((n>0)&&m>0);

    edges_.resize(n*m);
    filters_.resize(n*m);
    waiter_.set(n*m);

    real eta  = opts.optional_as<real>("eta", 0.0001);
    real mom  = opts.optional_as<real>("momentum", 0.0);
    real wd   = opts.optional_as<real>("weight_decay", 0.0);
    auto sz   = opts.require_as<ovec3i>("size");

    // filter size
    size_ = sz;

    // shared filter
    bool is_shared = false;
    bool need_init = true;
    if ( opts.contains("shared") )
    {
        auto name = opts.require_as<std::string>("shared");
        if ( filter::shared_filters_pool.count(name) == 0 )
        {
            auto& shared = filter::shared_filters_pool[name];
            shared.resize(n*m);
            for ( size_t k = 0; k < n*m; ++k )
            {
                shared[k] = std::make_shared<filter>(sz, eta, mom, wd);
            }
        }
        else
        {
            need_init = false;
        }
        filters_  = filter::shared_filters_pool[name];
        is_shared = true;
    }
    else
    {
        for ( size_t k = 0; k < n*m; ++k )
        {
            filters_[k] = std::make_shared<filter>(sz, eta, mom, wd);
        }
    }

    ZI_ASSERT(is_shared||need_init);

    if ( need_init )
    {
        std::string filter_values;

        if ( opts.contains("filters") )
        {
            filter_values = opts.require_as<std::string>("filters");
        }
        else
        {
            size_t n_coeffs = size_[0]*size_[1]*size_[2];
            size_t n_values = n*m*n_coeffs;
            real * filters_raw = new real[n_values];

            // additional information for initialization
            // e.g. fan-in, fan-out
            options info;
            info.push("fan-in",n*n_coeffs);
            info.push("fan-out",m*n_coeffs);

            auto initf = get_initializator( opts, &info );

            initf->initialize( filters_raw, n_values );

            filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                         sizeof(real) * n_values );
            delete [] filters_raw;
        }

        load_filters(filters_, size_, filter_values);
    }

    int does_fft = options_.optional_as<int>("fft", "0");
    auto repeat  = options_.optional_as<ovec3i>("repeat", "1,1,1");

    if ( size_ == vec3i::one ) does_fft = 0;

    for ( size_t i = 0, k = 0; i < n; ++i )
    {
        for ( size_t j = 0; j < m; ++j, ++k )
        {
            if ( repeat == ovec3i::one )
            {
                if ( does_fft )
                {
                    edges_[k]
                        = std::make_unique<fft_filter_edge>
                        (in, i, out, j, tm_, stride, *filters_[k], is_shared);
                }
                else
                {
                    edges_[k]
                        = std::make_unique<filter_edge>
                        (in, i, out, j, tm_, stride, *filters_[k], deconv,
                            is_shared);
                }
            }
            else
            {
                if ( does_fft )
                {
                    edges_[k]
                        = std::make_unique<fft_filter_ds_edge>
                        (in, i, out, j, tm_, stride, repeat, *filters_[k],
                            is_shared);
                }
                else
                {
                    edges_[k]
                        = std::make_unique<filter_ds_edge>
                        (in, i, out, j, tm_, stride, repeat, *filters_[k],
                            deconv, is_shared);
                }
            }
        }
    }
}

// dummy
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     dummy_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i] = std::make_unique<dummy_edge>
            (in, i, out, i, tm);
    }
}

// softmax
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     softmax_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto fwd = opts.optional_as<bool>("fwd",false);

    auto layer = std::make_shared<softmax_edge::layer>(n,tm);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i] = std::make_unique<softmax_edge>
            (in, i, out, i, tm, layer, fwd);
    }
}

// max-pooling (max-filtering + sparse convolution)
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     vec3i const & stride,
                     task_manager & tm,
                     max_pooling_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto sz = opts.require_as<ovec3i>("size");

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i]
            = std::make_unique<max_pooling_edge>
            (in, i, out, i, tm_, sz, stride);
    }
}

// max-pooling (subsampling)
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     real_pooling_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto sz = opts.require_as<ovec3i>("size");

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i]
            = std::make_unique<real_pooling_edge>
            (in, i, out, i, tm_, sz);
    }
}

// dropout
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     phase phs,
                     dropout_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto ratio = opts.optional_as<real>("ratio", 0.5);
    auto force = opts.optional_as<bool>("force", false);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i]
            = std::make_unique<dropout_edge>
            (in, i, out, i, tm_, ratio, phs, force);
    }
}

// nodeout
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     phase phs,
                     nodeout_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto ratio = opts.optional_as<real>("ratio", 0.5);
    auto force = opts.optional_as<bool>("force", false);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i]
            = std::make_unique<nodeout_edge>
            (in, i, out, i, tm_, ratio, phs, force);
    }
}

// crop
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     crop_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto offset = opts.require_as<ovec3i>("offset");

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i]
            = std::make_unique<crop_edge>
            (in, i, out, i, tm_, offset);
    }
}

// concat
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     concat_tag )
    : options_(opts)
    , tm_(tm)
{
    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto offset = opts.require_as<size_t>("offset");
    ZI_ASSERT(offset+n<=out->num_in_nodes());

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i]
            = std::make_unique<dummy_edge>
            (in, i, out, i+offset, tm_);
    }
}

// split
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     split_tag )
    : options_(opts)
    , tm_(tm)
{
    size_t n = out->num_in_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto offset = opts.require_as<size_t>("offset");
    ZI_ASSERT(offset+n<=in->num_out_nodes());

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i]
            = std::make_unique<dummy_edge>
            (in, i+offset, out, i, tm_);
    }
}

// maxout
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     maxout_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i] = std::make_unique<maxout_edge>
            (in, i, out, i, tm);
    }
}

// multiply
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     multiply_tag )
    : options_(opts)
    , tm_(tm)
{
    size_t n = in->num_out_nodes();
    size_t m = out->num_in_nodes();

    ZI_ASSERT(n==m||n==1);

    edges_.resize(m);
    waiter_.set(m);

    auto eps = opts.optional_as<real>("eps", 1e-5f);

    for ( size_t i = 0; i < m; ++i )
    {
        size_t inn = (n==1) ? 0 : i;
        edges_[i] = std::make_unique<multiply_edge>
            (in, inn, out, i, tm, eps);
    }
}

// normalize
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     phase phs,
                     normalize_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();

    edges_.resize(n);
    filters_.resize(n);
    waiter_.set(n);

    auto force = opts.optional_as<std::string>("force", "");
    auto frac  = opts.optional_as<real>("frac", 0.9999);
    auto eps   = opts.optional_as<real>("eps", 1e-5f);

    // TODO(lee):
    //      Each normalize edge has three real values to save/load.
    //      This is only a temporary solution that could have been done
    //      more neatly, but this is the best workaround as of now.
    const size_t num_vars = 3;
    auto sz = vec3i(1,1,num_vars);

    size_ = sz;

    for ( size_t i = 0; i < n; ++i )
    {
        filters_[i] = std::make_shared<filter>(sz,0,0,0);
    }

    std::string filter_values;

    if ( opts.contains("filters") )
    {
        filter_values = opts.require_as<std::string>("filters");
    }
    else
    {
        real * filters_raw = new real[n*num_vars];

        std::fill_n( filters_raw, n*num_vars, 0 );

        filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                     sizeof(real) * n * num_vars );
        delete [] filters_raw;
    }

    load_filters(filters_, size_, filter_values);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i] = std::make_unique<normalize_edge>
            (in, i, out, i, tm, force, frac, eps, *filters_[i], phs);
    }
}

// L2 norm
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     L2_norm_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();
    edges_.resize(n);
    waiter_.set(n);

    auto layer = std::make_shared<L2_norm_edge::layer>(n,tm);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i] = std::make_unique<L2_norm_edge>
            (in, i, out, i, tm, layer);
    }
}

// scale
inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     task_manager & tm,
                     scale_tag )
    : options_(opts)
    , tm_(tm)
{
    ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

    size_t n = in->num_out_nodes();

    edges_.resize(n);
    filters_.resize(n);
    waiter_.set(n);

    real eta  = opts.optional_as<real>("eta", 0.0001);
    real mom  = opts.optional_as<real>("momentum", 0.0);
    real wd   = opts.optional_as<real>("weight_decay", 0.0);
    auto sz   = vec3i::one;

    // filter size
    size_ = sz;

    for ( size_t i = 0; i < n; ++i )
    {
        filters_[i] = std::make_shared<filter>(sz, eta, mom, wd);
    }

    std::string filter_values;

    if ( opts.contains("filters") )
    {
        filter_values = opts.require_as<std::string>("filters");
    }
    else
    {
        real * filters_raw = new real[n];

        if ( opts.contains("init") )
        {
            options info;
            info.push("fan-in",1);
            info.push("fan-out",1);

            auto initf = get_initializator( opts, &info );

            initf->initialize( filters_raw, n );
        }
        else
        {
            std::fill_n( filters_raw, n, 1 );
        }

        filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                     sizeof(real) * n );
        delete [] filters_raw;
    }

    load_filters(filters_, size_, filter_values);

    for ( size_t i = 0; i < n; ++i )
    {
        edges_[i] = std::make_unique<filter_edge>
                        (in, i, out, i, tm_, vec3i::one, *filters_[i]);
    }
}

}}} // namespace znn::v4::parallel_network
