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

namespace znn { namespace v4 { namespace flow_graph {

class counter
{
private:
    std::size_t current_  = 0;
    std::size_t required_ = 0;

public:
    counter( std::size_t n = 0 )
        : required_(n)
    {}

    bool tick()
    {
        bool is_done = false;

        if ( ++current_ == required_ )
        {
            is_done  = true;
            current_ = 0;
        }

        return is_done;
    }

    void reset( std::size_t n )
    {
        current_  = 0;
        required_ = n;
    }

}; // class counter

}}} // namespace znn::v4::flow_graph