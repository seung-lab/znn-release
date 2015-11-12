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

#include "../../assert.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"
#include "../../cube/cube_operators.hpp"

namespace znn { namespace v4 {

inline std::vector<cube_p<real>>
make_affinity( cube<int> const & vol, size_t dim = 3 )
{
    // only allows 2D or 3D affinity graph
    ZI_ASSERT((dim==2)||(dim==3));

    vec3i s = size(vol);

    auto xaff = get_cube<real>(s); fill(*xaff,0);
    auto yaff = get_cube<real>(s); fill(*yaff,0);
    auto zaff = get_cube<real>(s); fill(*zaff,0);

    real one  = static_cast<real>(1);

    for ( size_t z = 0; z < s[0]; ++z )
        for ( size_t y = 0; y < s[1]; ++y )
            for ( size_t x = 0; x < s[2]; ++x )
            {
                int id = vol[z][y][x];

                if ( x > 0 && id > 0 )
                {
                    if ( id == vol[z][y][x-1] )
                        (*xaff)[z][y][x] = one;
                }

                if ( y > 0 && id > 0 )
                {
                    if ( id == vol[z][y-1][x] )
                        (*yaff)[z][y][x] = one;
                }

                if ( z > 0 && id > 0 && dim == 3 )
                {
                    if ( id == vol[z-1][y][x] )
                        (*zaff)[z][y][x] = one;
                }
            }

    std::vector<cube_p<real>> ret;
    ret.push_back(xaff);
    ret.push_back(yaff);
    ret.push_back(zaff);
    return ret;
}

inline std::vector<cube_p<real>>
make_affinity( cube<real> const & vol, size_t dim = 3 )
{
    // only allows 2D or 3D affinity graph
    ZI_ASSERT((dim==2)||(dim==3));

    vec3i s = size(vol);

    auto xaff = get_cube<real>(s); fill(*xaff,0);
    auto yaff = get_cube<real>(s); fill(*yaff,0);
    auto zaff = get_cube<real>(s); fill(*zaff,0);

    real one  = static_cast<real>(1);
    real zero = static_cast<real>(0);

    for ( size_t z = 0; z < s[0]; ++z )
        for ( size_t y = 0; y < s[1]; ++y )
            for ( size_t x = 0; x < s[2]; ++x )
            {
                if ( x > 0 )
                {
                    real v = std::min(vol[z][y][x-1],vol[z][y][x]);
                    (*xaff)[z][y][x] = std::max(std::min(v,one),zero);
                }

                if ( y > 0 )
                {
                    real v = std::min(vol[z][y-1][x],vol[z][y][x]);
                    (*yaff)[z][y][x] = std::max(std::min(v,one),zero);
                }

                if ( (z > 0) && (dim == 3) )
                {
                    real v = std::min(vol[z-1][y][x],vol[z][y][x]);
                    (*zaff)[z][y][x] = std::max(std::min(v,one),zero);
                }
            }

    std::vector<cube_p<real>> ret;
    ret.push_back(xaff);
    ret.push_back(yaff);
    ret.push_back(zaff);
    return ret;
}

}} // namespace znn::v4