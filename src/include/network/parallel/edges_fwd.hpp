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
    struct max_pooling_tag {};
    struct real_pooling_tag {};
    struct dropout_tag {};
    struct crop_tag {};
    struct softmax_tag {};
    struct maxout_tag {};

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

    edges( nodes *, nodes *, options const &, vec3i const &,
           task_manager &, phase phs, dropout_tag );

    edges( nodes *, nodes *, options const &, task_manager &, crop_tag );

    edges( nodes *, nodes *, options const &, task_manager &, softmax_tag );

    edges( nodes *, nodes *, options const &, task_manager &, maxout_tag );

    std::string name() const
    {
        return options_.require_as<std::string>("name");
    }

    // [kisuklee]
    // This is only temporary implementation and will be removed.
    void set_phase( phase phs )
    {
        for ( auto & e: edges_ )
        {
            e->set_phase(phs);
        }
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

    void set_patch_size( real s )
    {
        if ( filters_.size() )
        {
            for ( auto & e: edges_ ) e->set_patch_size(s);
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
