//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
//                    2015  Kisuk Lee           <kisuklee@mit.edu>
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

#include "../../assert.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"
#include "../../cube/cube_operators.hpp"
#include "zalis_def.hpp"
#include "get_segmentation.hpp"

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <zi/disjoint_sets/disjoint_sets.hpp>

namespace znn { namespace v4 {

inline zalis_weight
zalis( std::vector<cube_p<real>> true_affs,
       std::vector<cube_p<real>> affs,
       real high = 0.99,
       real low  = 0.01 )
{
    ZI_ASSERT(affs.size()==3);
    ZI_ASSERT(true_affs.size()==affs.size());

    double loss = 0;

    cube_p<int> seg_ptr = get_segmentation(true_affs);
    cube<int> const & seg = *seg_ptr;

    vec3i  s = size(*affs[0]);
    size_t n = s[0]*s[1]*s[2];

    cube<real> const & xaff = *affs[0];
    cube<real> const & yaff = *affs[1];
    cube<real> const & zaff = *affs[2];

    // merger weight
    std::vector<cube_p<real>> mw;
    auto xmw = get_cube<real>(s); fill(*xmw,0); mw.push_back(xmw);
    auto ymw = get_cube<real>(s); fill(*ymw,0); mw.push_back(ymw);
    auto zmw = get_cube<real>(s); fill(*zmw,0); mw.push_back(zmw);

    // splitter weight
    std::vector<cube_p<real>> sw;
    auto xsw = get_cube<real>(s); fill(*xsw,0); sw.push_back(xsw);
    auto ysw = get_cube<real>(s); fill(*ysw,0); sw.push_back(ysw);
    auto zsw = get_cube<real>(s); fill(*zsw,0); sw.push_back(zsw);

    // data structures for computing merger & splitter weight
    zi::disjoint_sets<uint32_t>     sets(n+1);
    std::vector<uint32_t>           sizes(n+1);
    std::map<uint32_t, uint32_t>    seg_sizes;
    std::vector<std::map<uint32_t, uint32_t> > contains(n+1);

    cube_p<int> ids_ptr = get_cube<int>(s);
    cube<int> & ids = *ids_ptr;

    for ( size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
        contains[i+1][seg.data()[i]] = 1;

        // non-boundary
        if ( seg.data()[i] > 0 )
        {
            ++seg_sizes[seg.data()[i]];
        }
    }

    // (affinity value, vertex 1, vertex 2, merger weight, splitter weight)
    typedef boost::tuple<real,uint32_t,uint32_t,real*,real*> edge_type;
    typedef std::greater<edge_type> edge_compare;

    std::vector<edge_type> edges;
    edges.reserve(n*3);

    for ( size_t z = 0; z < s[0]; ++z )
        for ( size_t y = 0; y < s[1]; ++y )
            for ( size_t x = 0; x < s[2]; ++x )
            {
                if ( x > 0 )
                {
                    real affinity = std::min(xaff[z][y][x],high);

                    if ( affinity > low )
                    {
                        edges.push_back(edge_type(affinity, ids[z][y][x-1],
                                              ids[z][y][x], &((*xmw)[z][y][x]),
                                              &((*xsw)[z][y][x])));
                    }
                }

                if ( y > 0 )
                {
                    real affinity = std::min(yaff[z][y][x],high);

                    if ( affinity > low )
                    {
                        edges.push_back(edge_type(affinity, ids[z][y-1][x],
                                              ids[z][y][x], &((*ymw)[z][y][x]),
                                              &((*ysw)[z][y][x])));
                    }
                }

                if ( z > 0 )
                {
                    real affinity = std::min(zaff[z][y][x],high);

                    if ( affinity > low )
                    {
                        edges.push_back(edge_type(affinity, ids[z-1][y][x],
                                              ids[z][y][x], &((*zmw)[z][y][x]),
                                              &((*zsw)[z][y][x])));
                    }
                }
            }

    std::sort(edges.begin(), edges.end(), edge_compare());

#if defined( DEBUG )
    std::cout << "\n[# edges] = " << edges.size() << std::endl;
    std::cout << '\n';

    auto ws = get_cube<int>(s);
    for ( size_t z = 0; z < s[0]; ++z )
        for ( size_t y = 0; y < s[1]; ++y )
            for ( size_t x = 0; x < s[2]; ++x )
                (*ws)[z][y][x] = sets.find_set(ids[z][y][x]);

    std::vector<cube_p<int>> ws_vec;
    ws_vec.push_back(ws);
#endif

    for ( auto& e: edges )
    {
        uint32_t set1 = sets.find_set(e.get<1>()); // watershed domain A
        uint32_t set2 = sets.find_set(e.get<2>()); // watershed domain B

        if ( set1 != set2 )
        {
            size_t n_same_pair = 0;
            size_t n_diff_pair = sizes[set1] * sizes[set2];

            for ( auto& contain: contains[set1] )
            {
                uint32_t segID   = contain.first;
                uint32_t segsize = contain.second;

                // boundary in A
                if ( segID == 0 )
                {
                    size_t pairs = segsize * sizes[set2];
                    n_diff_pair -= pairs;
                }
                else
                {
                    // same segments in both A and B
                    if ( contains[set2].find(segID) != contains[set2].end() )
                    {
                        size_t pairs = segsize * contains[set2][segID];
                        n_diff_pair -= pairs;
                        n_same_pair += pairs;
                    }

                    // boundary in B
                    if ( contains[set2].find(0) != contains[set2].end() )
                    {
                        size_t pairs = segsize * contains[set2][0];
                        n_diff_pair -= pairs;
                    }
                }
            }

            real* pmw = e.get<3>(); // merger weight
            real* psw = e.get<4>(); // splitter weight

            *pmw += n_diff_pair;
            *psw += n_same_pair;

#if defined( DEBUG )
            bool is_singleton = (sizes[set1] == 1) || (sizes[set2] == 1);
#endif

            uint32_t new_set = sets.join(set1, set2);
            sizes[set1] += sizes[set2];
            sizes[set2] = 0;

            for ( auto& contain: contains[set2] )
            {
                contains[set1][contain.first] += contain.second;
            }
            contains[set2].clear();

            std::swap(sizes[new_set], sizes[set1]);
            std::swap(contains[new_set], contains[set1]);

#if defined( DEBUG )
            // if ( !is_singleton )
            if ( true )
            {
                auto ws = get_cube<int>(s);
                for ( size_t z = 0; z < s[0]; ++z )
                    for ( size_t y = 0; y < s[1]; ++y )
                        for ( size_t x = 0; x < s[2]; ++x )
                            (*ws)[z][y][x] = sets.find_set(ids[z][y][x]);

                ws_vec.push_back(ws);
            }
#endif
        }
    }

    zalis_weight ret(mw,sw);
#if defined( DEBUG )
    ret.evolution = ws_vec;
    std::cout << "[evolution step] = " << ws_vec.size() << std::endl;
#endif

    return ret;
}


}} // namespace znn::v4
