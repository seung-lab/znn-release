//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
//                             Kisuk Lee           <kisuklee@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_INITIALIZATOR_NORMALIZE_INIT_HPP_INCLUDED
#define ZNN_CORE_INITIALIZATOR_NORMALIZE_INIT_HPP_INCLUDED

#include "initializator.hpp"

namespace zi {
namespace znn {

class normalize_init: public zinitializator
{
private:
    double lower_;
    double upper_;

    void do_initialize( vol<double>& v ) noexcept override
    {
        double min_val = min(v);
        double max_val = max(v);

        double old_range = max_val - min_val;
        double new_range = upper_  - lower_;

        if ( old_range < std::numeric_limits<double>::epsilon() )
        {
            old_range = std::numeric_limits<double>::max();
        }

        for ( std::size_t i = 0; i < v.num_elements(); ++i )
        {
            v.data()[i] =
                new_range * (v.data()[i] - min_val) / old_range + lower_;
        }

    }

public:
    explicit normalize_init( double low = 0, double up = 1 )
        : lower_(low)
        , upper_(up)
    {}

}; // class normalize_init

}} // namespace zi::znn

#endif // ZNN_CORE_INITIALIZATOR_NORMALIZE_INIT_HPP_INCLUDED
