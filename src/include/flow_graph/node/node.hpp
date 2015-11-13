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

namespace znn { namespace v4 { namespace flow_graph {

typedef std::vector<cube_p<real>>           tensor_type   ;
typedef std::map<std::string, tensor_type>  interface_type;

// Base node
class node
{
private:
    class counter
    {
    private:
        size_t current_  = 0;
        size_t required_ = 0;

    public:
        counter( size_t n = 0 )
            : required_(n)
        {}

        bool tick()
        {
            bool is_done = false;

            if ( ++current_ == required_ )
            {
                is_done  = true;
                current_ = 0;
            }

            return is_done;
        }

        void reset( size_t n )
        {
            current_  = 0;
            required_ = n;
        }
    };

protected:
    typedef std::map<std::string, node*>    targets_type;

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
            auto& targets = targets_[name];
            auto& tensor  = i.second;

            for ( auto& target: targets )
                target->receive(name, tensor);
        }
    }

protected:
    node( options const & op )
        : options_(op)
    {}

    options &       opts()       { return options_; }
    options const & opts() const { return options_; }

    interface_type & tops()    { return tops_;  }
    interface_type & bottoms() { return btms_;  }

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

    virtual void fwd_dispatch() { dispatch(tops_); }
    virtual void bwd_dispatch() { dispatch(btms_); }

    virtual void receive( std::string const & name, tensor const & T )
    {
        bool matched = false;

        // forward flow from bottom node
        if ( !matched && btms_.count(name) != 0 )
        {
            btms_[name] = T;
            if ( btms_counter_.tick() )
            {
                forward(btms_);
            }
            matched = true;
        }

        // backward flow from top node
        if ( !matched && tops_.count(name) != 0 )
        {
            tops_[name] = T;
            if ( tops_counter_.tick() )
            {
                backward(tops_);
            }
            matched = true;
        }

        ZI_ASSERT(matched);
    }

// define forward & backward computations
protected:
    virtual void setup()                      = 0;
    virtual void forward( interface_type & )  = 0;
    virtual void backward( interface_type & ) = 0;

public:
    std::string name() const
    {
        return options_.require_as<std::string>("name");
    }

    virtual bool is_bidirectional() const
    {
        return false;
    }

    virtual void add_target( std::string const & name, node* n )
    {
        ZI_ASSERT(targets_.count(name)==0);
        targets_[name] = n;
    }

public:
    virtual ~node() {}

};

}}} // namespace znn::v4