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
#include "zalis_def.hpp"

namespace znn { namespace v4 {

inline std::vector<cube_p<real>>
constrain_affinity( std::vector<cube_p<real>> const & true_affs,
                    std::vector<cube_p<real>> const & affs,
                    zalis_phase phase,
                    real threshold = 0.5 )
{
    ZI_ASSERT(true_affs.size()==affs.size());
    ZI_ASSERT(phase!=zalis_phase::BOTH);

    std::vector<cube_p<real>> constrained_affs;

    for ( size_t i = 0; i < true_affs.size(); ++i )
    {
        cube<real> const & taff = *true_affs[i];

        vec3i s = size(taff);

        constrained_affs.push_back(get_cube<real>(s));
        cube<real>& aff = *constrained_affs.back();
        aff = *affs[i];

        ZI_ASSERT(size(taff)==size(aff));

        for ( size_t z = 0; z < s[0]; ++z )
            for ( size_t y = 0; y < s[1]; ++y )
                for ( size_t x = 0; x < s[2]; ++x )
                {
                    // constrain merger to boundary
                    if ( phase == zalis_phase::MERGER )
                    {
                        if ( taff[z][y][x] > threshold )
                        {
                            aff[z][y][x] = taff[z][y][x];
                        }
                    }

                    // constrain splitter to non-boundary
                    if ( phase == zalis_phase::SPLITTER )
                    {
                        if ( taff[z][y][x] < threshold )
                        {
                            aff[z][y][x] = taff[z][y][x];
                        }
                    }
                }
    }

    return constrained_affs;
}

}} // namespace znn::v4