//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZI_ZNN_CORE_TASK_MANAGER_HPP_INCLUDED
#define ZI_ZNN_CORE_TASK_MANAGER_HPP_INCLUDED

#include <functional>
#include <thread>
#include <map>
#include <list>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <limits>

#include <zi/utility/singleton.hpp>

namespace zi {
namespace znn {

class task_manager
{
private:
    typedef std::function<void()> callable_t;

private:
    std::map<std::size_t, std::list<callable_t*>> tasks_;

    std::size_t spawned_threads_;
    std::size_t concurrency_    ;
    std::size_t idle_threads_   ;

    std::mutex              mutex_;
    std::condition_variable manager_cv_;
    std::condition_variable workers_cv_;

private:
    void worker_loop()
    {
        {
            std::lock_guard<std::mutex> g(mutex_);

            if ( spawned_threads_ >= concurrency_ )
            {
                return;
            }

            ++spawned_threads_;
            if ( spawned_threads_ == concurrency_ )
            {
                manager_cv_.notify_all();
            }
        }

        while (true)
        {
            callable_t* f = nullptr;

            {
                std::unique_lock<std::mutex> g(mutex_);

                while ( tasks_.empty() && concurrency_ >= spawned_threads_ )
                {
                    ++idle_threads_;
                    workers_cv_.wait(g);
                    --idle_threads_;
                }

                if ( tasks_.empty() )
                {
                    --spawned_threads_;
                    if ( spawned_threads_ == concurrency_ )
                    {
                        manager_cv_.notify_all();
                    }
                    return;
                }

                f = next_task();
            }

            (*f)();
            delete f;;
        }
    }

public:
    task_manager(std::size_t concurrency = std::thread::hardware_concurrency())
        : spawned_threads_{0}
        , concurrency_{0}
        , idle_threads_{0}
    {
        set_concurrency(concurrency);
    }

    task_manager(const task_manager&) = delete;
    task_manager& operator=(const task_manager&) = delete;

    task_manager(task_manager&& other) = delete;
    task_manager& operator=(task_manager&&) = delete;

    ~task_manager()
    {
        set_concurrency(0);
    }

    std::size_t set_concurrency(std::size_t n)
    {
        std::unique_lock<std::mutex> g(mutex_);

        if ( concurrency_ != spawned_threads_ )
        {
            return concurrency_;
        }

        std::size_t to_spawn = (n > concurrency_) ? ( n - concurrency_ ) : 0;
        concurrency_ = n;

        for ( std::size_t i = 0; i < to_spawn; ++i )
        {
            std::thread t(&task_manager::worker_loop, this);
            t.detach();
        }

        workers_cv_.notify_all();

        while ( concurrency_ != spawned_threads_ )
        {
            manager_cv_.wait(g);
        }

        return concurrency_;
    }

    std::size_t get_concurrency()
    {
        std::lock_guard<std::mutex> g(mutex_);
        return concurrency_;
    }

    std::size_t idle_threads()
    {
        std::lock_guard<std::mutex> g(mutex_);
        return idle_threads_;
    }

    std::size_t active_threads()
    {
        std::lock_guard<std::mutex> g(mutex_);
        return concurrency_ - idle_threads_;
    }

private:
    callable_t* next_task()
    {
        callable_t* f = tasks_.rbegin()->second.front();

        tasks_.rbegin()->second.pop_front();
        if ( tasks_.rbegin()->second.size() == 0 )
        {
            tasks_.erase(tasks_.rbegin()->first);
        }
        return f;
    }

public:
    template<typename... Args>
    void schedule(std::size_t priority, Args&&... args)
    {
        callable_t* fn = new callable_t(std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            tasks_[priority].emplace_front(fn);
            if ( idle_threads_ > 0 ) workers_cv_.notify_all();
        }
    }

    template<typename... Args>
    void schedule_asap(Args&&... args)
    {
        schedule(std::numeric_limits<std::size_t>::max(),
                 std::forward<Args>(args)...);
    }

    template<typename... Args>
    void schedule_eventually(Args&&... args)
    {
        schedule(0, std::forward<Args>(args)...);
    }

}; // class task_manager


}} // namespace zi::znn


#endif //ZI_ZNN_CORE_TASK_MANAGER_HPP_INCLUDED
