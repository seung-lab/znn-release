#pragma once

#include "types.hpp"

#include <map>
#include <utility>
#include <iostream>

namespace znn { namespace v4 {

class options: std::map<std::string,std::string>
{
public:
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
            std::cout << p.first << '=' << p.second << '\n';
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


}} // namespace znn::v4
