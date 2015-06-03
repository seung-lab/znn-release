#pragma once

#include "edges_fwd.hpp"
#include "filter_edge.hpp"
#include "fft_filter_edge.hpp"
#include "filter_ds_edge.hpp"
#include "fft_filter_ds_edge.hpp"
#include "dummy_edge.hpp"
#include "max_pooling_edge.hpp"
#include "nodes.hpp"
#include "../../utils/waiter.hpp"
#include "../../options/options.hpp"
#include "../filter.hpp"


namespace znn { namespace v4 { namespace parallel_network {

inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     vec3i const & stride,
                     vec3i const & in_size,
                     task_manager & tm,
                     filter_tag )
    : options_(opts)
    , size_(in_size)
    , tm_(tm)
{
    size_t n = in->num_out_nodes();
    size_t m = out->num_in_nodes();

    ZI_ASSERT((n>0)&&m>0);

    edges_.resize(n*m);
    filters_.resize(n*m);
    waiter_.set(n*m);

    real eta  = opts.optional_as<real>("eta", 0.1);
    real mom  = opts.optional_as<real>("momentum", 0.0);
    real wd   = opts.optional_as<real>("weight_decay", 0.0);
    auto   sz   = opts.require_as<ovec3i>("size");

    size_ = sz;

    for ( size_t k = 0; k < n*m; ++k )
    {
        filters_[k] = std::make_unique<filter>(sz, eta, mom, wd);
    }

    std::string filter_values;


    if ( opts.contains("filters") )
    {
        filter_values = opts.require_as<std::string>("filters");
    }
    else
    {
        size_t n_values = n*m*size_[0]*size_[1]*size_[2];
        real * filters_raw = new real[n_values];

        auto initf = get_initializator(opts);


        initf->initialize( filters_raw, n*m*size_[0]*size_[1]*size_[2] );

        filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                     sizeof(real) * n_values );
        delete [] filters_raw;
    }

    load_filters(filters_, size_, filter_values);

    int does_fft = options_.optional_as<int>("fft", "1");
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
                        (in, i, out, j, tm_, stride, *filters_[k]);
                }
                else
                {
                    edges_[k]
                        = std::make_unique<filter_edge>
                        (in, i, out, j, tm_, stride, *filters_[k]);
                }
            }
            else
            {
                if ( does_fft )
                {
                    edges_[k]
                        = std::make_unique<fft_filter_ds_edge>
                        (in, i, out, j, tm_, stride, repeat, *filters_[k]);
                }
                else
                {
                    edges_[k]
                        = std::make_unique<filter_ds_edge>
                        (in, i, out, j, tm_, stride, repeat, *filters_[k]);
                }
            }
        }
    }



}

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


inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     vec3i const & stride,
                     vec3i const & in_size,
                     task_manager & tm,
                     max_pooling_tag )
    : options_(opts)
    , size_(in_size)
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

inline edges::edges( nodes * in,
                     nodes * out,
                     options const & opts,
                     vec3i const & in_size,
                     task_manager & tm,
                     real_pooling_tag )
    : options_(opts)
    , size_(in_size)
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


}}} // namespace znn::v4::parallel_network
