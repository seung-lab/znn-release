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

#include "../../cube/cube.hpp"

namespace znn { namespace v4 {

enum class zalis_phase : std::uint8_t {BOTH = 0, MERGER = 1, SPLITTER = 2};

struct zalis_weight
{
    std::vector<cube_p<real>>   merger;
    std::vector<cube_p<real>>   splitter;
    real rand_error;  // rand error
    real num_non_bdr; // number of non-boundary voxels
    real TP;
    real TN;
    real FP;
    real FN;

#if defined( DEBUG )
    std::vector<cube_p<int>>    ws_snapshots;
    std::vector<int>            ws_timestamp;
    std::vector<cube_p<int>>    timestamp;
#endif

    zalis_weight(std::vector<cube_p<real>> m,
                 std::vector<cube_p<real>> s,
                 real re,
                 real n,
                 real tp,
                 real tn,
                 real fp,
                 real fn)
        : merger(m)
        , splitter(s)
        , rand_error( re )
        , num_non_bdr( n )
        , TP(tp)
        , TN(tn)
        , FP(fp)
        , FN(fn)
#if defined( DEBUG )
        , ws_snapshots()
        , ws_timestamp()
        , timestamp()
#endif
    {}
};

}} // namespace znn::v4
