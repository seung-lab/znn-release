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

#ifndef ZNN_MALIS2_HPP_INCLUDED
#define ZNN_MALIS2_HPP_INCLUDED

#include "malis.hpp"

namespace zi {
namespace znn {

inline malis_pair
malis2( std::list<double3d_ptr> true_affs,
        std::list<double3d_ptr> affs,
        std::list<bool3d_ptr>   masks,
        double margin, bool pos, bool neg )
{
    double loss = 0;
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

    zi::disjoint_sets<uint32_t>     sets(n+1);
    std::vector<uint32_t>           sizes(n+1);
    std::map<uint32_t, uint32_t>    seg_sizes;
    std::vector<std::map<uint32_t, uint32_t> > contains(n+1);


    long3d_ptr ids_ptr = volume_pool.get_long3d(s);
    long3d& ids = *ids_ptr;

    for ( std::size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
        contains[i+1][seg.data()[i]] = 1;

        // non-boundary
        if ( seg.data()[i] != 0 )
        {
            ++seg_sizes[seg.data()[i]];
        }
    }

    // normalization factor
    double frac_pos = 0;    
    FOR_EACH( it, seg_sizes )
    {
        frac_pos += (it->second - static_cast<double>(1)) /
                        (static_cast<double>(2)*it->second);
    }
    double frac_neg = (seg_sizes.size()*(seg_sizes.size()-1))/2;    
    double frac_norm = 0;
    if ( pos ) frac_norm += frac_pos;
    if ( neg ) frac_norm += frac_neg;

    // 3D Rand error
    double nTP = 0;
    double nFP = 0;
    double nFN = 0;
    double nTN = 0;
    
    typedef boost::tuple<double,uint32_t,uint32_t,double*> edge_type;
    typedef std::greater<edge_type>                        edge_compare;

    std::vector<edge_type> edges;
    edges.reserve(n*3);

    for ( std::size_t x = 0; x < s[0]; ++x )
        for ( std::size_t y = 0; y < s[1]; ++y )
            for ( std::size_t z = 0; z < s[2]; ++z )
            {
                if ( x > 0 )
                {
                    if ( (*xmask)[x][y][z] )
                        edges.push_back(edge_type((*xaff)[x][y][z], ids[x-1][y][z],
                                                  ids[x][y][z], &((*xres)[x][y][z])));
                }

                if ( y > 0 )
                {
                    if ( (*ymask)[x][y][z] )
                        edges.push_back(edge_type((*yaff)[x][y][z], ids[x][y-1][z],
                                                  ids[x][y][z], &((*yres)[x][y][z])));
                }

                if ( z > 0 )
                {
                    if ( (*zmask)[x][y][z] )
                        edges.push_back(edge_type((*zaff)[x][y][z], ids[x][y][z-1],
                                                  ids[x][y][z], &((*zres)[x][y][z])));
                }
            }

    std::sort(edges.begin(), edges.end(), edge_compare());
    
    FOR_EACH( it, edges )
    {
        uint32_t set1 = sets.find_set(it->get<1>()); // region A
        uint32_t set2 = sets.find_set(it->get<2>()); // region B

        if ( set1 != set2 )
        {
            double n_frac_same = 0;
            double n_frac_diff = 0;

            FOR_EACH( sit1, contains[set1] )
            {
                std::size_t id1 = sit1->first;

                // skip boundary
                if ( id1 == 0 ) continue;

                FOR_EACH( sit2, contains[set2] )
                {
                    std::size_t id2 = sit2->first;

                    // skip boundary
                    if ( id2 == 0 ) continue;

                    double frac = sit1->second * sit2->second;
                    frac /= seg_sizes[id1] * seg_sizes[id2];

                    if ( id1 == id2 )
                    {
                        n_frac_same += frac;
                    }
                    else
                    {
                        n_frac_diff += frac;
                    }
                }
            }

            if ( (it->get<0>()) > 0.5 )
            {
                nTP += n_frac_same;
                nFP += n_frac_diff;
            }
            else
            {
                nTN += n_frac_diff;
                nFN += n_frac_same;
            }

            double* p = it->get<3>();

            // hinge loss
            {
                // delta(s_i,s_j) = 1
                if ( pos )
                {
                    double l = std::max(0.0,margin-(it->get<0>()-0.5));
                    *p   -= (l > 0)*n_frac_same;
                    loss += l*n_frac_same;
                }

                // delta(s_i,s_j) = 0
                if ( neg )
                {
                    double l = std::max(0.0,margin+(it->get<0>()-0.5));
                    *p   += (l > 0)*n_frac_diff;
                    loss += l*n_frac_diff;
                }
            }

            // normlize gradient
            *p /= frac_norm;

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
    
    malis_metric metric;
    metric.loss = loss/frac_norm;
    metric.nTP  = nTP/frac_norm;
    metric.nFP  = nFP/frac_norm;
    metric.nFN  = nFN/frac_norm;
    metric.nTN  = nTN/frac_norm;

    return std::make_pair(affs, metric);
}


}} // namespace zi::znn

#endif // ZNN_MALIS2_HPP_INCLUDED
