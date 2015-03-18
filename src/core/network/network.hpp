//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
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

#ifndef ZNN_CORE_NETWORK_NETWORK_HPP_INCLUDED
#define ZNN_CORE_NETWORK_NETWORK_HPP_INCLUDED

#include "../types.hpp"
#include "../utils.hpp"

#include <map>
#include <string>

namespace zi { namespace znn {

struct nodes;

struct edges
{
    vec3i width     = vec3i::one  ;
    vec3i stride    = vec3i::one  ;

    vec3i in_stride = vec3i::zero ;
    vec3i in_fsize  = vec3i::zero ;

    nodes* in;
    nodes* out;

    explicit edges( const vec3i& _width = vec3i::one,
                    const vec3i& _stride = vec3i::one )
        : width(_width), stride(_stride)
    {}
};

struct nodes
{
    vec3i fov       = vec3i::zero;
    vec3i stride    = vec3i::zero;
    vec3i fsize     = vec3i::zero;

    std::vector<edges*> in, out;
};

class network
{
private:
    enum class state_t { uninitialized, running, stopped };
    enum class pass_t  { none, forward, backward         };

    std::map<std::string, edges*> edges_;
    std::map<std::string, nodes*> nodes_;
    std::map<std::string, nodes*> input_nodes_;
    std::map<std::string, nodes*> output_nodes_;

private:
    network(const network&) = delete;
    network& operator=(const network&) = delete;

    network(network&&) = delete;
    network& operator=(network&&) = delete;

private:
    void fov_pass(nodes* n, const vec3i& fov, const vec3i& fsize )
    {
        if ( n->fov != vec3i::zero )
        {
            ZI_ASSERT(n->fsize==fsize);
            ZI_ASSERT(n->fov==fov);
        }
        else
        {
            for ( auto& e: n->out )
            {
                e->in_fsize = fsize;
            }
            n->fov = fov;
            n->fsize = fsize;
            for ( auto& e: n->in )
            {
                vec3i new_fov   = (fov - vec3i::one) * e->stride + e->width;
                vec3i new_fsize = (e->width-vec3i::one) * e->in_stride + fsize;
                fov_pass(e->in, new_fov, new_fsize);
            }
        }
    }

    void stride_pass(nodes* n, const vec3i& stride )
    {
        if ( n->stride != vec3i::zero )
        {
            ZI_ASSERT(n->stride==stride);
        }
        else
        {
            n->stride = stride;
            for ( auto& e: n->out )
            {
                e->in_stride = stride;
                stride_pass(e->out, stride * e->stride );
            }
        }
    }


public:
    network() {}

public:

    void init()
    {
        for ( auto& o: input_nodes_ )
            stride_pass(o.second, vec3i::one);
        for ( auto& o: output_nodes_ )
            fov_pass(o.second, vec3i::one, vec3i{1,1,1});

        for ( auto& o: nodes_ )
        {
            std::cout << o.first << ' ' << o.second->fov
                      << ' ' << o.second->stride
                      << ' ' << o.second->fsize << '\n';
        }

        for ( auto& o: edges_ )
        {
            std::cout << o.first << ' ' << o.second->width
                      << ' ' << o.second->stride
                      << ' ' << o.second->in_stride << '\n';
        }

    }

    void add_edges(const std::string& name, edges* e,
                   const std::string& in, const std::string& out)
    {
        ZI_ASSERT(edges_.count(name)==0);
        ZI_ASSERT(nodes_.count(in)&&nodes_.count(out));
        edges_[name] = e;
        e->in  = nodes_[in];
        e->out = nodes_[out];
        nodes_[in]->out.push_back(e);
        nodes_[out]->in.push_back(e);
    }

    void add_nodes(const std::string& name, nodes* n)
    {
        ZI_ASSERT(nodes_.count(name)==0);
        nodes_[name] = n;
    }

    void add_input_nodes(const std::string& name, nodes* n)
    {
        ZI_ASSERT(nodes_.count(name)==0);
        input_nodes_[name] = n;
        nodes_[name] = n;
    }

    void add_output_nodes(const std::string& name, nodes* n)
    {
        ZI_ASSERT(nodes_.count(name)==0);
        output_nodes_[name] = n;
        nodes_[name] = n;
    }


};

}} // namespace zi::znn

#endif // ZNN_CORE_NETWORK_NETWORK_HPP_INCLUDED
