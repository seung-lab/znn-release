//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_DESCRIPTION_DESCRIPTION_HPP_INCLUDED
#define ZNN_CORE_DESCRIPTION_DESCRIPTION_HPP_INCLUDED

#include "types.hpp"
#include <string>
#include <iostream>
#include <map>
#include <boost/lexical_cast.hpp>

namespace zi {
namespace znn {

class descriptor: std::map<std::string,std::string>
{
public:
    template<typename V>
    descriptor& push(const std::string& k, const V& v)
    {
        (*this)[k] = boost::lexical_cast<std::string>(v);
        return *this;
    }

    void dump()
    {
        for ( auto& p: (*this) )
        {
            std::cout << p.first << '=' << p.second << '\n';
        }
    }

    template<typename T>
    T as(const std::string& k)
    {
        return boost::lexical_cast<T>((*this)[k]);
    }

    template<typename T>
    T optional_as(const std::string& k, const T& v)
    {
        if ( this->count(k) )
            return boost::lexical_cast<T>((*this)[k]);
        else
            return v;
    }

    template<typename T>
    T optional_as(const std::string& k, const std::string& v)
    {
        if ( this->count(k) )
            return boost::lexical_cast<T>((*this)[k]);
        else
            return boost::lexical_cast<T>(v);
    }

    template<typename T>
    T require_as(const std::string& k)
    {
        if ( this->count(k) )
            return boost::lexical_cast<T>((*this)[k]);
        else
            throw std::logic_error("missing parameter: " + k);
    }

};


}} // namespace zi::znn


#endif // ZNN_CORE_DESCRIPTION_DESCRIPTION_HPP_INCLUDED
