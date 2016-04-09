//
// Copyright (C)      2016  Kisuk Lee           <kisuklee@mit.edu>
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

#include <map>

namespace znn { namespace v4 {

template <typename T>
using sample = std::map<std::string, tensor<T>>;

template <typename T>
class dataset
{
public:
    // draw a next sample in a random sequence
    virtual sample<T> random_sample() = 0;

    // draw a next sample in a predetermined sequence
    virtual sample<T> next_sample() = 0;

protected:
    dataset() {}

public:
    virtual ~dataset() {}

}; // class dataset

}} // namespace znn::v4