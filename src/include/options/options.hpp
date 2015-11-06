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

#include "types.hpp"

#include <map>
#include <initializer_list>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>

namespace znn { namespace v4 {

class options: public std::map<std::string,std::string>
{
public:

    options() {}

    options( std::initializer_list<std::pair<std::string,std::string>> l)
    {
        for ( auto & p: l )
        {
            (*this)[boost::lexical_cast<std::string>(p.first)]
                = boost::lexical_cast<std::string>(p.second);
        }
    }

    template<typename V>
    options& push(const std::string& k, const V& v)
    {
        (*this)[k] = boost::lexical_cast<std::string>(v);
        return *this;
    }

    options& push(const options& other)
    {
        for ( auto& x: other )
        {
            (*this)[x.first] = x.second;
        }
        return *this;
    }

    options& push(options&& other)
    {
        for ( auto& x: other )
        {
            (*this)[x.first] = x.second;
        }
        return *this;
    }

    void dump() const
    {
        for ( auto& p: (*this) )
        {
            if ( p.first != "filters" && p.first != "biases" )
            {
                std::cout << p.first << '=' << p.second << '\n';
            }
            else
            {
                std::cout << p.first << '=' << "[binary]" << '\n';
            }
        }
    }

    template<typename T>
    T as(const std::string& k)
    {
        return boost::lexical_cast<T>((*this)[k]);
    }

    template<typename T>
    T optional_as(const std::string& k, const T& v) const
    {
        auto x = this->find(k);
        if ( x != this->end() )
            return boost::lexical_cast<T>(x->second);
        else
            return v;
    }

    template<typename T>
    T optional_as(const std::string& k, const std::string& v) const
    {
        auto x = this->find(k);
        if ( x != this->end() )
            return boost::lexical_cast<T>(x->second);
        else
            return boost::lexical_cast<T>(v);
    }

    template<typename T>
    T require_as(const std::string& k) const
    {
        auto x = this->find(k);
        if ( x != this->end() )
            return boost::lexical_cast<T>(x->second);
        else
            throw std::logic_error("missing parameter: " + k);
    }

    bool contains( std::string const & k ) const
    {
        return this->count(k);
    }

    template< class CharT, class Traits >
    friend ::std::basic_ostream< CharT, Traits >&
    operator<<( ::std::basic_ostream< CharT, Traits >& os, const options& op )
    {
        for ( auto& p: op )
        {
            os << p.first << '=' << p.second << '\n';
        }
        return os;
    }

};

void parse_net_file( std::vector<options> & nodes,
                     std::vector<options> & edges,
                     std::string const & fname )
{
    nodes.clear();
    edges.clear();

    std::vector<options>* current;

    std::string line;
    std::ifstream ifs(fname.c_str());

    while ( std::getline(ifs, line) )
    {
        if ( line.size() )
        {
            std::istringstream iss(line);
            std::string a, b;
            iss >> a >> b;
            if ( a == "nodes" )
            {
                current = & nodes;
                current->resize(current->size()+1);
                current->back().push("name", b);
            }
            else if ( a == "edges" )
            {
                current = & edges;
                current->resize(current->size()+1);
                current->back().push("name", b);
            }
            else
            {
                STRONG_ASSERT(current);
                current->back().push(a,b);
            }
        }
    }
}

void parse_option_file( options & op, std::string const & fname )
{
    op.clear();

    std::string line;
    std::ifstream ifs(fname.c_str());

    while ( std::getline(ifs, line) )
    {
        if ( line.size() )
        {
            std::istringstream iss(line);
            std::string a, b;
            iss >> a >> b;
            op.push(a,b);
        }
    }
}


}} // namespace znn::v4
