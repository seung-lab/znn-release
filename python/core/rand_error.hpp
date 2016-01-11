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

#include "assert.hpp"
#include "types.hpp"
#include "cube/cube.hpp"
#include "cube/cube_operators.hpp"
#include "flow_graph/computation/zalis_def.hpp"
#include "flow_graph/computation/get_segmentation.hpp"

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <zi/disjoint_sets/disjoint_sets.hpp>

namespace znn { namespace v4 {

inline real
get_rand_error( std::vector<cube_p<real>> affs,
                std::vector<cube_p<real>> true_affs )
{
    ZI_ASSERT(affs.size()==3);
    ZI_ASSERT(true_affs.size()==affs.size());

    cube_p<int> seg_ptr = get_segmentation(true_affs);
    cube<int> const & seg = *seg_ptr;

    vec3i  s = size(*affs[0]);
    size_t n = s[0]*s[1]*s[2];


#if defined( DEBUG )
    std::cout<<"\n \n \n prop affinity: "<<std::endl;
    cube<real> const & affx = *affs[0];
    for(size_t i=0; i<100; i++)
        std::cout<<affx.data()[i]<<",";

    std::cout<<"\n true affinity: "<<std::endl;
    cube<real> const & taffx = *true_affs[0];
    for(size_t i=0; i<100; i++)
        std::cout<<taffx.data()[i]<<",";
    std::cout<<"\n segmentation: "<<std::endl;
    for(size_t i=0; i<100; i++)
        std::cout<<seg.data()[i]<<",";
#endif

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

#if defined( DEBUG )
    // timestamp
    std::vector<cube_p<int>> ts;
    auto xts = get_cube<int>(s); fill(*xts,0); ts.push_back(xts);
    auto yts = get_cube<int>(s); fill(*yts,0); ts.push_back(yts);
    auto zts = get_cube<int>(s); fill(*zts,0); ts.push_back(zts);
#endif

    // data structures for computing merger & splitter weight
    zi::disjoint_sets<uint32_t>     sets(n+1);
    std::vector<uint32_t>           sizes(n+1);
    std::map<uint32_t, uint32_t>    seg_sizes;
    std::vector<std::map<uint32_t, uint32_t> > contains(n+1);

    cube_p<int> ids_ptr = get_cube<int>(s);
    cube<int> & ids = *ids_ptr;

    // initialize the counting
    real FP=0;
    real FN=0;
    real TP=0;
    real TN=0;
    real num_non_bdr=0;

    for ( size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
        contains[i+1][seg.data()[i]] = 1;

        // non-boundary
        if ( seg.data()[i] > 0 )
        {
	    ++num_non_bdr;
            ++seg_sizes[seg.data()[i]];
        }
    }

#if defined( DEBUG )
    typedef boost::tuple<real,uint32_t,uint32_t,real*,real*,int*> edge_type;
#else
    // (affinity value, vertex 1, vertex 2, merger weight, splitter weight)
    typedef boost::tuple<real,uint32_t,uint32_t,real*,real*> edge_type;
#endif
    typedef std::greater<edge_type> edge_compare;

    std::vector<edge_type> edges;
    edges.reserve(n*3);

    for ( size_t z = 0; z < s[0]; ++z )
        for ( size_t y = 0; y < s[1]; ++y )
            for ( size_t x = 0; x < s[2]; ++x )
            {
                if ( x > 0 )
                {
                    real affinity = xaff[z][y][x];
#if defined( DEBUG )
                    edges.push_back(edge_type(affinity, ids[z][y][x-1],
                                              ids[z][y][x], &((*xmw)[z][y][x]),
                                              &((*xsw)[z][y][x]),
                                              &((*xts)[z][y][x])));
#else
                    edges.push_back(edge_type(affinity, ids[z][y][x-1],
                                              ids[z][y][x], &((*xmw)[z][y][x]),
                                              &((*xsw)[z][y][x])));
#endif

                }

                if ( y > 0 )
                {
                    real affinity = yaff[z][y][x];

#if defined( DEBUG )
                    edges.push_back(edge_type(affinity, ids[z][y-1][x],
                                              ids[z][y][x], &((*ymw)[z][y][x]),
                                              &((*ysw)[z][y][x]),
                                              &((*yts)[z][y][x])));
#else
                    edges.push_back(edge_type(affinity, ids[z][y-1][x],
                                              ids[z][y][x], &((*ymw)[z][y][x]),
                                              &((*ysw)[z][y][x])));
#endif
                }

                if ( z > 0 )
                {
                    real affinity = zaff[z][y][x];


#if defined( DEBUG )
                    edges.push_back(edge_type(affinity, ids[z-1][y][x],
                                              ids[z][y][x], &((*zmw)[z][y][x]),
                                              &((*zsw)[z][y][x]),
                                              &((*zts)[z][y][x])));
#else
                    edges.push_back(edge_type(affinity, ids[z-1][y][x],
                                              ids[z][y][x], &((*zmw)[z][y][x]),
                                              &((*zsw)[z][y][x])));
#endif
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

    std::vector<cube_p<int>> ws_snapshots;
    ws_snapshots.push_back(ws);

    size_t timestamp = 0;
    std::vector<int> ws_timestamp;
    ws_timestamp.push_back(timestamp);
#endif

    for ( auto& e: edges )
    {
        real affinity = e.get<0>();                // edge affinity
        uint32_t set1 = sets.find_set(e.get<1>()); // watershed domain A
        uint32_t set2 = sets.find_set(e.get<2>()); // watershed domain B

        if ( set1 != set2 )
        {
            real n_same_pair = 0;
            real n_diff_pair = 0;

            // each segment fraction in watershed domain A
            for ( auto& seg1: contains[set1] )
            {
                uint32_t segID1   = seg1.first;
                real     segsize1 = seg1.second;

                // skip boundary
                if ( segID1 == 0 ) continue;

                // each segment fraction in watershed domain B
                for ( auto& seg2: contains[set2] )
                {
                    uint32_t segID2   = seg2.first;
                    real     segsize2 = seg2.second;

                    // skip boundary
                    if ( segID2 == 0 ) continue;

                    // a pair of fractions belongs to the same segment
                    if ( segID1 == segID2 )
                    {
                        n_same_pair += segsize1 * segsize2;
                        if (affinity < 0.5)
                            FN += segsize1 * segsize2;
                        else
                            TP += segsize1 * segsize2;
                    }
                    else
                    {
                        n_diff_pair += segsize1 * segsize2;
                        if (affinity >0.5)
                            FP += segsize1 * segsize2;
                        else
                            TN += segsize1 * segsize2;
                    }
                }
            }

            real* pmw = e.get<3>(); // merger weight
            real* psw = e.get<4>(); // splitter weight

            *pmw += n_diff_pair;
            *psw += n_same_pair;

#if defined( DEBUG )
            bool is_singleton = (sizes[set1] == 1) || (sizes[set2] == 1);

            // // before merging non-singleton watershed domains
            // if ( !is_singleton && timestamp > ws_timestamp.back() )
            // {
            //     auto ws = get_cube<int>(s);
            //     for ( size_t z = 0; z < s[0]; ++z )
            //         for ( size_t y = 0; y < s[1]; ++y )
            //             for ( size_t x = 0; x < s[2]; ++x )
            //                 (*ws)[z][y][x] = sets.find_set(ids[z][y][x]);

            //     ws_snapshots.push_back(ws);
            //     ws_timestamp.push_back(timestamp);
            // }
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
            *(e.get<5>()) = ++timestamp; // timestamp

            // after merging non-singleton watershed domains
            if ( !is_singleton )
            {
                auto ws = get_cube<int>(s);
                for ( size_t z = 0; z < s[0]; ++z )
                    for ( size_t y = 0; y < s[1]; ++y )
                        for ( size_t x = 0; x < s[2]; ++x )
                            (*ws)[z][y][x] = sets.find_set(ids[z][y][x]);

                ws_snapshots.push_back(ws);
                ws_timestamp.push_back(timestamp);
            }
#endif
        }
    }

    // rand error
    if(num_non_bdr<=1)
    {
        std::cout<<"\n num of non-boundary pixels: "<<num_non_bdr<<std::endl;
        std::cout<<"FP: "<<FP<<", FN: "<<FN<<", TP: "<<TP<<", TN: "<<TN<<std::endl;
    }
    real re = (FP+FN) / (num_non_bdr*(num_non_bdr-1)/2);
    ZI_ASSERT( num_non_bdr*(num_non_bdr-1)/2==TP+FN+TN+FP );
    return re;
}


}} // namespace znn::v4
