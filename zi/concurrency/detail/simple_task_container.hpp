//
// Copyright (C) 2010  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
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

#ifndef ZI_CONCURRENCY_DETAIL_SIMPLE_TASK_CONTAINER_HPP
#define ZI_CONCURRENCY_DETAIL_SIMPLE_TASK_CONTAINER_HPP 1

#include <zi/concurrency/runnable.hpp>
#include <zi/bits/shared_ptr.hpp>
#include <deque>

namespace zi {
namespace concurrency_ {
namespace detail {

class simple_task_container: public std::deque< shared_ptr<runnable> >
{
private:
    typedef std::deque< shared_ptr<runnable> > super_type;

public:
    shared_ptr<runnable> get_pop_front()
    {
        shared_ptr<runnable> ret = super_type::front();
        super_type::pop_front();
        return ret;
    }
};

} // namespace detail
} // namespace concurrency_
} // namespace zi

#endif

