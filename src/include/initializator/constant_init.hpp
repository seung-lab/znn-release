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

class constant_init: public initializator<real>
{
private:
    real c_;

    void do_initialize( real*v, size_t n ) noexcept override
    {
        std::fill_n(v, n, c_);
    }

public:
    explicit constant_init( real c = 0 ): c_(c) {}

}; // class constant_init

}} // namespace znn::v4
