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

#include <functional>
#include <thread>
#include <fstream>
#include <atomic>
#include <map>
#include <set>
#include <list>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <utility>

#include <zi/utility/singleton.hpp>
#include <zi/time.hpp>

#include "log.hpp"

namespace znn { namespace v4 {

class global_task_manager_impl
{
private:
    typedef std::function<void()> callable_t;

private:
    std::list <callable_t*> tasks_;

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
            callable_t* f1 = nullptr;

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

                f1 = next_task();
            }

            (*f1)();
            delete f1;
        }
    }


public:
    global_task_manager_impl(std::size_t concurrency = 240)
        : spawned_threads_{0}
        , concurrency_{0}
        , idle_threads_{0}
    {
        set_concurrency(concurrency);
    }

    global_task_manager_impl(const global_task_manager_impl&) = delete;
    global_task_manager_impl& operator=(const global_task_manager_impl&) = delete;

    global_task_manager_impl(global_task_manager_impl&& other) = delete;
    global_task_manager_impl& operator=(global_task_manager_impl&&) = delete;

    ~global_task_manager_impl()
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
            std::thread t(&global_task_manager_impl::worker_loop, this);
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
        callable_t* f = tasks_.front();
        tasks_.pop_front();
        return f;
    }

public:
    template<typename... Args>
    void schedule(Args&&... args)
    {
        callable_t* fn = new callable_t(std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            tasks_.push_front(fn);
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
        }
    }


}; // class global_task_manager_impl

namespace {
global_task_manager_impl& global_task_manager =
    zi::singleton<global_task_manager_impl>::instance();
}


}} // namespace znn::v4
