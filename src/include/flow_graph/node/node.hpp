//
// Copyright (C)      2015  Kisuk Lee           <kisuklee@mit.edu>
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

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../options/options.hpp"
#include "../utils/counter.hpp"

namespace znn { namespace v4 { namespace flow_graph {


typedef std::vector<cube_p<real>>           tensor_type   ;
typedef std::map<std::string, tensor_type>  interface_type;


class node_base
{
private:
    // [TODO]
    // Diverging forward?
    // Diverging backward?
    typedef std::map<std::string, std::vector<node_base*>> targets_type;

private:
    options      options_;

    // tensors
    interface_type  tops_;
    interface_type  btms_;

    // targets
    targets_type targets_;

    // counters
    counter tops_counter_;
    counter btms_counter_;

private:
    bool load( interface_type & interface,
               std::string const & name,
               tensor_type && tensor )
    {
        bool loaded = false;

        if ( interface.count(name) != 0 )
        {
            interface[name] = std::move(tensor);
            loaded = true;
        }

        return loaded;
    }

    void load( interface_type &  target,
               interface_type && source )
    {
        for ( auto& i: source )
            if ( target.count(i.first) != 0 )
                target[name] = std::move(i.second);
    }

    void dispatch( interface_type & interface )
    {
        for ( auto& i: interface )
        {
            auto& name    = i.first;
            auto& tensor  = i.second;
            auto& targets = targets_[name];

            // always pass-by-copy, considering diverging flows
            for ( auto& target: targets )
                target->receive(name, tensor);
        }
    }

// non-copyable
private:
    node_base( node_base const & ) = delete;
    node_base& operator=( node_base const & ) = delete;

    node_base( node_base && ) = delete;
    node_base& operator=( node_base && ) = delete;


protected:
    explicit node_base( options const & op )
        : options_(op)
    {
        // add tops
        auto tops = op.require_as<ovector<std::string>>("tops");
        for ( auto& top: tops )

        // add bottoms
        auto btms = op.require_as<ovector<std::string>>("bottoms");
        for ( auto& btm: btms )
    }

    options &       opts()       { return options_; }
    options const & opts() const { return options_; }

    interface_type  tops()       { return tops_; }
    interface_type  bottoms()    { return btms_; }


// Load a "gun" (interface) before "firing" (flow)
protected:
    void fwd_load( std::string const & name, tensor_type && tensor )
    {
        load(tops_, name, std::move(tensor));
    }

    void bwd_load( std::string const & name, tensor_type && tensor )
    {
        load(tops_, name, std::move(tensor));
    }

    void fwd_load( interface_type && interface )
    {
        load(tops_, std::move(interface));
    }

    void bwd_load( interface_type && interface )
    {
        load(btms_, std::move(interface));
    }


// Core communication functions between nodes
protected:
    void fwd_dispatch() { dispatch(tops_); }
    void bwd_dispatch() { dispatch(btms_); }

    void receive( std::string const & name, tensor const & T )
    {
        bool matched = false;

        // forward flow from a bottom node
        if ( !matched && btms_.count(name) != 0 )
        {
            btms_[name] = T;
            if ( btms_counter_.tick() )
            {
                forward();
            }
            matched = true;
        }

        // backward flow from a top node
        if ( !matched && tops_.count(name) != 0 )
        {
            tops_[name] = T;
            if ( tops_counter_.tick() )
            {
                backward();
            }
            matched = true;
        }

        ZI_ASSERT(matched);
    }


// Define graph-internal forward & backward computations.
protected:
    virtual void forward()  = 0;
    virtual void backward() = 0;

// Define graph-external forward & backward computations.
// Except for loop-allowed nodes (e.g. interface node),
// it should return the result of computation defined by the node itself,
// i.e., flowless self-computation.
public:
    virtual interface_type forward( interface_type && in )   = 0;
    virtual interface_type backward( interface_type && out ) = 0;

public:
    virtual bool is_bidirectional() const = 0;
    virtual void setup() = 0;

public:
    std::string name() const
    {
        return options_.require_as<std::string>("name");
    }

    void add_top( std::string const & name, node_base* target )
    {
        // add top
        static_cast<void>(tops_[name]);

        // add forward target
        ZI_ASSERT(target);
        targets_[name].push_back(target);
    }

    void add_bottom( std::string const & name, node_base* target = nullptr )
    {
        // add bottom
        static_cast<void>(btms_[name]);

        // add backward target (optional)
        if ( !target )
        {
            ZI_ASSERT(target->is_bidirectional());
            targets_[name].push_back(target);
        }
    }

public:
    virtual ~node_base() {}

}; // abstract class node_base


template< bool Bidirectional >
class node: public node_base
{
public:
    bool is_bidirectional() const override
    {
        return Bidirectional;
    }

protected:
    explicit node( options const & op )
        : node_base(op)
    {}

public:
    virtual ~node() {}

}; // abstract class template node


typedef node<true>  bidirectional_node ;
typedef node<false> unidirectional_node;


}}} // namespace znn::v4::flow_graph