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

#include "../initializator/initializators.hpp"

#include <map>
#include <iostream>

namespace znn { namespace v4 {

typedef std::vector<std::map<std::string, std::vector<cube_p<real>>>> inout_t;

template<typename Net>
inline std::pair<inout_t,inout_t> generate_inout( size_t n, Net const & net )
{
    auto ins  = net.inputs();
    auto outs = net.outputs();

    uniform_init init(1);
    std::pair<inout_t, inout_t> ret;

    std::cout << "Generating inputs/outputs\n";

    for ( auto & ip: ins )
    {
        for ( size_t i = 0; i < ip.second.second; ++i )
        {
            std::cout << "\tINPUT " << ip.first << "["
                      << i << "] " << ip.second.first << "\n";
        }
    }

    for ( auto & ip: outs )
    {
        for ( size_t i = 0; i < ip.second.second; ++i )
        {
            std::cout << "\tOUTPUT " << ip.first << "["
                      << i << "] " << ip.second.first << "\n";
        }
    }



    for ( size_t i = 0; i < n; ++i )
    {
        std::map<std::string, std::vector<cube_p<real>>> insample;
        std::map<std::string, std::vector<cube_p<real>>> outsample;

        for ( auto & ip: ins )
        {
            for ( size_t i = 0; i < ip.second.second; ++i )
            {
                auto v = get_cube<real>(ip.second.first);
                init.initialize(*v);
                insample[ip.first].push_back(v);
            }
        }

        ret.first.push_back(insample);

        for ( auto & ip: outs )
        {
            for ( size_t i = 0; i < ip.second.second; ++i )
            {
                auto v = get_cube<real>(ip.second.first);
                init.initialize(*v);
                outsample[ip.first].push_back(v);
            }
        }

        ret.second.push_back(outsample);
    }
    return ret;
}

template<typename C>
std::pair<double, double> measured( C const & c )
{
    for ( auto a: c ) std::cout << a << "\n";
    double mean   = std::accumulate(c.begin(), c.end(), 0.0) / c.size();
    double sqsum  = std::inner_product(c.begin(), c.end(), c.begin(), 0.0);
    double stderr = std::sqrt((sqsum/c.size() - mean*mean) / c.size());
    return std::make_pair(mean, stderr);
}


}} // namespace znn::v4
