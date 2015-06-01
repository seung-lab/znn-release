#pragma once

#include "edge.hpp"
#include "nodes.hpp"
#include "../filter.hpp"
#include "../../options/options.hpp"
#include "../../utils/waiter.hpp"
#include "../../utils/task_manager.hpp"
#include "../../initializator/initializators.hpp"
#include "../trivial/utils.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class edges
{
public:
    struct filter_tag {};
    struct dummy_tag {};
    struct max_pooling_tag{};
    struct real_pooling_tag{};

protected:
    options                                options_;
    waiter                                 waiter_ ;
    std::vector<std::unique_ptr<edge>>     edges_  ;
    std::vector<std::unique_ptr<filter>>   filters_;
    vec3i                                  size_   ;
    task_manager &                         tm_     ;

public:
    edges( nodes *, nodes *, options const &, vec3i const &, vec3i const &,
           task_manager &, filter_tag );

    edges( nodes *, nodes *, options const &, task_manager &, dummy_tag );

    edges( nodes *, nodes *, options const &, vec3i const &, vec3i const &,
           task_manager &, max_pooling_tag );

    edges( nodes *, nodes *, options const &, vec3i const &,
           task_manager &, real_pooling_tag );

    std::string name() const
    {
        return options_.require_as<std::string>("name");
    }

    void set_eta( real eta )
    {
        if ( filters_.size() )
        {
            options_.push("eta", eta);
            for ( auto & f: filters_ ) f->eta() = eta;
        }
    }

    void set_momentum( real mom )
    {
        if ( filters_.size() )
        {
            options_.push("momentum", mom);
            for ( auto & f: filters_ ) f->momentum() = mom;
        }
    }

    void set_weight_decay( real wd )
    {
        if ( filters_.size() )
        {
            options_.push("weight_decay", wd);
            for ( auto & f: filters_ ) f->weight_decay() = wd;
        }
    }

    options serialize() const
    {
        options ret = options_;
        if ( filters_.size() )
        {
            ret.push("filters", save_filters(filters_, size_));
        }
        return ret;
    }

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
