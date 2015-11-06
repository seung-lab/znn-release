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

#include <zi/disjoint_sets/disjoint_sets.hpp>

namespace znn { namespace v4 {

inline cube_p<int>
get_segmentation( std::vector<cube_p<real>> affs,
                  real threshold = 0.5 )
{
    // only takes 3D affinity graph
    ZI_ASSERT(affs.size==3);

    vec3i  s = size(*affs[0]);
    size_t n = s[0]*s[1]*s[2];

    cube<real> const & xaff = *affs[0];
    cube<real> const & yaff = *affs[1];
    cube<real> const & zaff = *affs[2];

    zi::disjoint_sets<uint32_t> sets(n+1);
    std::vector<uint32_t>       sizes(n+1);

    cube_p<int> ids_ptr = get_cube<int>(s);
    cube<int> & ids     = *ids_ptr;

    for ( size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
    }

    typedef std::pair<uint32_t,uint32_t> edge_type;
    std::vector<edge_type> edges;

    // thresholded affinity graph
    for ( size_t z = 0; z < s[0]; ++z )
    {
        for ( size_t y = 0; y < s[1]; ++y )
        {
            for ( size_t x = 0; x < s[2]; ++x )
            {
                long id1 = ids[z][y][x];

                if ( x > 0 )
                {
                    // skip disconnected (black) edges
                    // only count connected (white) edges
                    if ( xaff[z][y][x] > threshold )
                    {
                        long id2 = ids[z][y][x-1];
                        edges.push_back(edge_type(id1, id2));
                    }
                }

                if ( y > 0 )
                {
                    // skip disconnected (black) edges
                    // only count connected (white) edges
                    if ( yaff[z][y][x] > threshold )
                    {
                        long id2 = ids[z][y-1][x];
                        edges.push_back(edge_type(id1, id2));
                    }
                }

                if ( z > 0 )
                {
                    // skip disconnected (black) edges
                    // only count connected (white) edges
                    if ( zaff[z][y][x] > threshold )
                    {
                        long id2 = ids[z-1][y][x];
                        edges.push_back(edge_type(id1, id2));
                    }
                }
            }
        }
    }

    // connected components of the thresholded affinity graph
    // computed the size of each connected component
    for ( auto& e: edges )
    {
        uint32_t set1 = sets.find_set(e.first);
        uint32_t set2 = sets.find_set(e.second);

        if ( set1 != set2 )
        {
            uint32_t new_set = sets.join(set1, set2);

            sizes[set1] += sizes[set2];
            sizes[set2]  = 0;

            std::swap(sizes[new_set], sizes[set1]);
        }
    }

    std::vector<uint32_t> remaps(n+1);

    uint32_t next_id = 1;

    // assign a unique segment ID to each of
    // the pixel in a connected component
    for ( size_t i = 0; i < n; ++i )
    {
        uint32_t id = sets.find_set(ids.data()[i]);
        if ( sizes[id] > 1 )
        {
            if ( remaps[id] == 0 )
            {
                remaps[id]    = next_id;
                ids.data()[i] = next_id;
                ++next_id;
            }
            else
            {
                ids.data()[i] = remaps[id];
            }
        }
        else
        {
            remaps[id]    = 0;
            ids.data()[i] = 0;
        }
    }

    return ids_ptr;
}

}} // namespace znn::v4
