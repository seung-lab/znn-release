//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_MERGER_HPP_INCLUDED
#define ZNN_MERGER_HPP_INCLUDED

#include "malis.hpp"

namespace zi {
namespace znn {

inline  std::list<double3d_ptr>
merger( std::list<double3d_ptr> true_affs,
        std::list<double3d_ptr> affs,
        std::list<bool3d_ptr>   masks,
<<<<<<< HEAD
        double margin = 0 )    
=======
        double margin )
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f
{
    long3d_ptr seg_ptr = get_segmentation(true_affs, masks);
    long3d&    seg = *seg_ptr;

    ZI_ASSERT(affs.size()==3);

    vec3i s = volume_utils::volume_size(affs.front());
    std::size_t n = s[0]*s[1]*s[2];

    double3d_ptr xaff = affs.front(); affs.pop_front();
    double3d_ptr yaff = affs.front(); affs.pop_front();
    double3d_ptr zaff = affs.front(); affs.pop_front();

    bool3d_ptr xmask = masks.front(); masks.pop_front();
    bool3d_ptr ymask = masks.front(); masks.pop_front();
    bool3d_ptr zmask = masks.front(); masks.pop_front();

    double3d_ptr xres = volume_pool.get_double3d(s); volume_utils::zero_out(xres);
    double3d_ptr yres = volume_pool.get_double3d(s); volume_utils::zero_out(yres);
    double3d_ptr zres = volume_pool.get_double3d(s); volume_utils::zero_out(zres);

    affs.push_back(xres);
    affs.push_back(yres);
    affs.push_back(zres);

    zi::disjoint_sets<uint32_t> sets(n+1);
    std::vector<uint32_t>       sizes(n+1);
    std::vector<std::map<uint32_t, uint32_t> > contains(n+1);


    long3d_ptr ids_ptr = volume_pool.get_long3d(s);
    long3d& ids = *ids_ptr;

    for ( std::size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
        contains[i+1][seg.data()[i]] = 1;
    }

<<<<<<< HEAD
    // std::size_t n_pairs = n*(n-1)/2;

    typedef boost::tuple<double,uint32_t,uint32_t,double*> edge_type;
    typedef std::greater<edge_type>                        edge_compare;
=======
    std::size_t n_pairs = n*(n-1)/2;

    typedef boost::tuple<uint32_t,uint32_t,double*> edge_type;
    typedef std::greater<edge_type>                 edge_compare;
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f

    std::vector<edge_type> edges;
    edges.reserve(n*3);

    for ( std::size_t x = 0; x < s[0]; ++x )
        for ( std::size_t y = 0; y < s[1]; ++y )
            for ( std::size_t z = 0; z < s[2]; ++z )
            {
                if ( x > 0 )
                {
                    if ( (*xmask)[x][y][z] )
<<<<<<< HEAD
                        edges.push_back(edge_type((*xaff)[x][y][z], ids[x-1][y][z],
                                                  ids[x][y][z], &((*xres)[x][y][z])));
=======
                        edges.push_back(edge_type(ids[x-1][y][z], ids[x][y][z], 
                                                  &((*xres)[x][y][z])));
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f
                }

                if ( y > 0 )
                {
                    if ( (*ymask)[x][y][z] )
<<<<<<< HEAD
                        edges.push_back(edge_type((*yaff)[x][y][z], ids[x][y-1][z],
                                                  ids[x][y][z], &((*yres)[x][y][z])));
=======
                        edges.push_back(edge_type(ids[x][y-1][z], ids[x][y][z],
                                                  &((*yres)[x][y][z])));
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f
                }

                if ( z > 0 )
                {
                    if ( (*zmask)[x][y][z] )
<<<<<<< HEAD
                        edges.push_back(edge_type((*zaff)[x][y][z], ids[x][y][z-1],
                                                  ids[x][y][z], &((*zres)[x][y][z])));
=======
                        edges.push_back(edge_type(ids[x][y][z-1], ids[x][y][z],
                                                  &((*zres)[x][y][z])));
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f
                }
            }

    std::sort(edges.begin(), edges.end(), edge_compare());

<<<<<<< HEAD
=======
    std::size_t incorrect = 0;

>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f
    // [kisuklee]
    // (B,N) or (B,B) pairs where B: boundary, N: non-boundary
    std::size_t n_b_pairs = 0;

    FOR_EACH( it, edges )
    {
<<<<<<< HEAD
        uint32_t set1 = sets.find_set(it->get<1>()); // region A
        uint32_t set2 = sets.find_set(it->get<2>()); // region B
=======
        uint32_t set1 = sets.find_set(it->get<0>()); // region A
        uint32_t set2 = sets.find_set(it->get<1>()); // region B
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f

        if ( set1 != set2 )
        {
            std::size_t n_pair_same = 0;
            std::size_t n_pair_diff = sizes[set1]*sizes[set2];

            FOR_EACH( sit, contains[set1] )
            {
                // boundary
                if ( sit->first == 0 )
                {
<<<<<<< HEAD
                    std::size_t pairs = sit->second * sizes[set2];
                    n_pair_diff -= pairs;
                    n_b_pairs   += pairs;
=======
                    FOR_EACH( sit2, contains[set2] )
                    {
                        std::size_t pairs = sit->second * sit2->second;
                        n_pair_diff -= pairs;
                        n_b_pairs   += pairs;
                    }
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f
                }
                else // non-boundary
                {
                    if ( contains[set2].find(sit->first) != contains[set2].end() )
                    {
                        std::size_t pairs = sit->second * contains[set2][sit->first];
                        n_pair_diff -= pairs;
                        n_pair_same += pairs;
                    }
<<<<<<< HEAD

                    if ( contains[set2].find(0) != contains[set2].end() )
                    {
                        std::size_t pairs = sit->second * contains[set2][0];
                        n_pair_diff -= pairs;
                        n_b_pairs   += pairs;
                    }
                }
            }

            double* p = it->get<3>();

            // delta(s_i,s_j) = 1
            double dl = -std::max(0.0,0.5+margin-(it->get<0>()));
            *p   += dl*n_pair_same;

            // delta(s_i,s_j) = 0
            dl = std::max(0.0,(it->get<0>())-0.5+margin);
            *p   += dl*n_pair_diff;

            // merger: 1, splitter: -1
            if ( *p > 0.5 )
            {
                *p = static_cast<double>(1);
            }
            else if ( *p < -0.5 )
            {
                *p = static_cast<double>(-1);
            }
=======
                }
            }

            double* p = it->get<2>();

            // delta(s_i,s_j) = 0
            *p = n_pair_diff > 0 ? static_cast<double>(1):static_cast<double>(0);
>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f

            uint32_t new_set = sets.join(set1, set2);
            sizes[set1] += sizes[set2];
            sizes[set2] = 0;

            FOR_EACH( sit, contains[set2] )
            {
                contains[set1][sit->first] += sit->second;
            }
            contains[set2].clear();

            std::swap(sizes[new_set], sizes[set1]);
            std::swap(contains[new_set], contains[set1]);
        }
    }

<<<<<<< HEAD
=======
    std::size_t n_eff_pairs = n_pairs - n_b_pairs;

>>>>>>> 005ef59c9b566c154be4ac5ae5625854339ee72f
    return affs;
}


}} // namespace zi::znn

#endif // ZNN_MERGER_HPP_INCLUDED
