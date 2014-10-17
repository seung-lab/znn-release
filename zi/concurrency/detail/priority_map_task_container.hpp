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

#ifndef ZI_CONCURRENCY_DETAIL_PRIORITY_MAP_TASK_CONTAINER_HPP
#define ZI_CONCURRENCY_DETAIL_PRIORITY_MAP_TASK_CONTAINER_HPP 1

#include <zi/concurrency/runnable.hpp>
#include <zi/bits/shared_ptr.hpp>
#include <zi/bits/cstdint.hpp>

#include <cstddef>
#include <utility>
#include <functional>
#include <set>

namespace zi {


namespace concurrency_ {
namespace detail {

struct priority_map_task_container
{
private:
    std::set< std::pair<std::size_t, shared_ptr< runnable > > > tasks_ ;

public:
    explicit priority_map_task_container(): tasks_()
    {
    }

    std::size_t size() const
    {
        return tasks_.size();
    }

    std::size_t empty() const
    {
        return tasks_.empty();
    }

    void clear()
    {
        tasks_.clear();
    }

    shared_ptr< runnable > front()
    {
        return tasks_.begin()->second;
    }

    shared_ptr< runnable > get_pop_front()
    {
        shared_ptr<runnable> ret = tasks_.begin()->second;
        tasks_.erase( tasks_.begin() );
        return ret;
    }

    void pop_front()
    {
        tasks_.erase( tasks_.begin() );
    }

    void insert( shared_ptr< runnable > task, std::size_t prio )
    {
        tasks_.insert(std::make_pair(prio, task));
    }

};


} // namespace detail
} // namespace concurrency_
} // namespace zi

#endif

