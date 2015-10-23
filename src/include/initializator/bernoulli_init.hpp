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

#include "initializator.hpp"

namespace znn { namespace v4 {

template <typename T>
class bernoulli_init: public initializator<T>
{
private:
    std::bernoulli_distribution dis;

    void do_initialize( T* v, size_t n ) noexcept override
    {
        initializator<T>::initialize_with_distribution(dis, v, n);
    }

public:
    explicit bernoulli_init( real p = 0.5 )
        : dis(p)
    {}

}; // class bernoulli_init

template class bernoulli_init<real>;
template class bernoulli_init<bool>;

}} // namespace znn::v4
