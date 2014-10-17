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

#ifndef ZI_CONCURRENCY_DETAIL_RANDOMIZED_TASK_CONTAINER_HPP
#define ZI_CONCURRENCY_DETAIL_RANDOMIZED_TASK_CONTAINER_HPP 1

#include <zi/concurrency/runnable.hpp>
#include <zi/bits/shared_ptr.hpp>
#include <zi/utility/assert.hpp>

#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <ctime>


namespace zi {

namespace concurrency_ {
namespace detail {

struct randomized_task_container
{
private:
    std::vector< shared_ptr< runnable > > tasks_;

public:
    explicit randomized_task_container()
        : tasks_()
    {
        std::srand(std::time(0));
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

    void push_front( shared_ptr< runnable > task )
    {
        tasks_.push_back(task);
    }

    void push_back( shared_ptr< runnable > task )
    {
        tasks_.push_back(task);
    }

    shared_ptr<runnable> get_pop_front()
    {
        ZI_ASSERT(tasks_.size()>0);

        const std::size_t idx = std::rand() % tasks_.size();
        shared_ptr<runnable> ret = tasks_[idx];

        tasks_[idx] = tasks_[tasks_.size()-1];
        tasks_.pop_back();

        return ret;
    }

};


} // namespace detail
} // namespace concurrency_
} // namespace zi

#endif

