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

#include "initializator.hpp"
#include <algorithm>

namespace znn { namespace v4 {

class normalize_init: public initializator<real>
{
private:
    real lower_;
    real upper_;

    void do_initialize( real* v, size_t n ) noexcept override
    {
        real min_val = *std::min_element(v,v+n);
        real max_val = *std::max_element(v,v+n);

        real old_range = max_val - min_val;
        real new_range = upper_  - lower_;

        if ( old_range < std::numeric_limits<real>::epsilon() )
        {
            old_range = std::numeric_limits<real>::max();
        }

        for ( std::size_t i = 0; i < n; ++i )
        {
            v[i] = new_range * (v[i] - min_val) / old_range + lower_;
        }

    }

public:
    explicit normalize_init( real low = 0, real up = 1 )
        : lower_(low)
        , upper_(up)
    {}

}; // class normalize_init

}} // namespace znn::v4
