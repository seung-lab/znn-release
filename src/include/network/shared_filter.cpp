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

#include <mutex>

#include "filter.hpp"

namespace znn { namespace v4 {

class shared_filter: public filter
{
private:
    std::mutex  mutex_;

public:
    shared_filter( const vec3i& s, real eta, real mom = 0.0, real wd = 0.0 )
        : filter(s,eta,mom,wd)
    {
    }

    virtual void update(const cube<real>& dEdW, real patch_size = 1 ) override
    {
        guard g(mutex_);

        filter::update(dEdW, patch_size);
    }

}; // class filter

}} // namespace znn::v4
