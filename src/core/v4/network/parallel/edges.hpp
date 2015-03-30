#pragma once

#include "edge.hpp"
#include "filter_edge.hpp"
#include "dummy_edge.hpp"
#include "max_pooling_edge.hpp"
#include "nodes.hpp"
#include "../../utils/waiter.hpp"
#include "../../options/options.hpp"
#include "../filter.hpp"


namespace znn { namespace v4 { namespace parallel_network {

class edges
{
public:
    struct filter_tag {};
    struct dummy_tag {};
    struct pooling_edges_tag{};

protected:
    options                                options_;
    waiter                                 waiter_ ;
    std::vector<std::unique_ptr<edge>>     edges_  ;
    std::vector<std::unique_ptr<filter>>   filters_;
    vec3i                                  size_   ;
    task_manager &                         tm_     ;

public:
    edges( nodes * in,
           nodes * out,
           options const & opts,
           vec3i const & stride,
           vec3i const & in_size,
           task_manager & tm,
           filter_tag )
    {
        size_t n = in->num_out_nodes();
        size_t m = out->num_in_nodes();

        ZI_ASSERT((n>0)&&m>0);

        edges_.resize(n*m);
        filters_.resize(n*m);
        waiter_.set(n*m);

        double eta    = opts.optional_as<double>("eta", 0.1);
        double mom    = opts.optional_as<double>("momentum", 0.0);
        double wd     = opts.optional_as<double>("weight_decay", 0.0);
        auto   sz     = opts.require_as<ovec3i>("size");

        size_ = sz;

        for ( size_t i = 0, k = 0; i < n; ++i )
        {
            for ( size_t j = 0; j < m; ++j, ++k )
            {
                filters_[k] = std::make_unique<filter>(sz, eta, mom, wd);
                edges_[k]
                    = std::make_unique<filter_edge>
                    (in, i, out, j, stride, *filters_[k], tm);
            }
        }

        std::string filter_values;

        opts.dump();

        if ( opts.contains("filters") )
        {
            filter_values = opts.require_as<std::string>("filters");
        }
        else
        {
            size_t n_values = n*m*size_[0]*size_[1]*size_[2];
            double * filters_raw = new double[n_values];

            auto initf = get_initializator(opts);


            initf->initialize( filters_raw, n*m*size_[0]*size_[1]*size_[2] );
            delete [] filters_raw;

            filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                         sizeof(double) * n_values );
        }

        load_filters(filters_, size_, filter_values);
    }

    virtual ~edges() {}

    virtual void set_eta( double )
    { UNIMPLEMENTED(); }

    virtual void set_momentum( double )
    { UNIMPLEMENTED(); }

    virtual void set_weight_decay( double )
    { UNIMPLEMENTED(); }

    virtual options serialize() const = 0;

    void edge_zapped()
    {
        waiter_.one_done();
    }

    void zap()
    {
        for ( auto & e: edges_ )
        {
            e->zap(this);
        }
        waiter_.wait();
    }
};

}}} // namespace znn::v4::parallel_network
