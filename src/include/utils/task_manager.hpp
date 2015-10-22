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
//#include <boost/pool/pool_alloc.hpp>

#include "log.hpp"
#include "global_task_manager.hpp"

#ifdef ZNN_DFS_TASK_SCHEDULER
#  include "dfs_task_manager.hpp"
#else

namespace znn { namespace v4 {

namespace detail {

class function_base
{
public:
    virtual ~function_base() {}
    virtual void operator()() = 0;
};

template<typename T>
class function_wrapper: function_base
{
private:
    T t_;
public:
    function_wrapper(T && t): t_(t) {};
    function_wrapper(T const &) = delete;
};



} // namespace detail

class task_manager
{
private:
    typedef std::function<void()> callable_t;

private:
    struct unprivileged_task
    {
    private:
        std::function<void()>  fn_               ;
        callable_t*            then_   = nullptr ;
        int                    status_ = 1       ;

        list<std::shared_ptr<unprivileged_task>>::iterator it_;

        friend class task_manager;

    public:
        template<class... Args>
        explicit unprivileged_task(Args&&... args)
            : fn_(std::bind(std::forward<Args>(args)...))
        {}
    };

public:
    typedef std::shared_ptr<unprivileged_task> task_handle;

private:
    map<std::size_t, list<callable_t*>> tasks_                  ;
    size_t                                       tot_tasks_ = 0 ;
    list<task_handle>                            unprivileged_  ;

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

        task_handle f2;

        while (true)
        {
            callable_t* f1 = nullptr;

            {
                std::unique_lock<std::mutex> g(mutex_);

                while ( tasks_.empty() &&
                        unprivileged_.empty() &&
                        concurrency_ >= spawned_threads_ )
                {
                    ++idle_threads_;
                    workers_cv_.wait(g);
                    --idle_threads_;
                }

                if ( tasks_.empty() && unprivileged_.empty() )
                {
                    --spawned_threads_;
                    if ( spawned_threads_ == concurrency_ )
                    {
                        manager_cv_.notify_all();
                    }
                    return;
                }

                if ( tasks_.size() )
                {
                    f1 = next_task();
                }
                else
                {
                    f1 = nullptr;
                    f2 = next_unprivileged_task();
                }

            }

            if ( f1 )
            {
                (*f1)();
                allocator<callable_t> alloc;
                alloc.destroy(f1);
                alloc.deallocate(f1,1);
            }
            else
            {
                execute_unprivileged_task(f2);
            }
        }
    }

private:
    // executing in one of the manaer's threads
    void execute_unprivileged_task(task_handle const & t)
    {
        t->fn_();
        t->fn_ = nullptr;

        callable_t* after = nullptr;

        {
            std::unique_lock<std::mutex> g(mutex_);
            after      = t->then_;
            t->status_ = 0;
        }

        if ( after )
        {
            (*after)();
            allocator<callable_t> alloc;
            alloc.destroy(after);
            alloc.deallocate(after,1);
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
            //std::thread t(&task_manager::worker_loop, this);
            //t.detach();
            global_task_manager.schedule(&task_manager::worker_loop, this);
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

        --tot_tasks_;
        // LOG(info) << tot_tasks_
        //           << ", " << unprivileged_.size() << ";";

        return f;
    }

    task_handle next_unprivileged_task()
    {
        auto x = unprivileged_.front();
        x->status_ = 2;
        unprivileged_.pop_front();
        return x;
    }

public:
    template<typename... Args>
    void schedule(std::size_t priority, Args&&... args)
    {
        allocator<callable_t> alloc;
        callable_t* fn = alloc.allocate(1);
        alloc.construct(fn, std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            tasks_[priority].emplace_front(fn);
            ++tot_tasks_;
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
        }
    }

    template<typename... Args>
    void asap(Args&&... args)
    {
        allocator<callable_t> alloc;
        callable_t* fn = alloc.allocate(1);
        alloc.construct(fn, std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            tasks_[std::numeric_limits<std::size_t>::max()].emplace_front(fn);
            ++tot_tasks_;
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
        }
    }

    template<typename... Args>
    void require_done(task_handle const & t, Args&&... args)
    {
        // doesn't exist!
        if ( !t )
        {
            std::bind(std::forward<Args>(args)...)();
            return;
        }

        bool stolen = false;

        {
            std::lock_guard<std::mutex> g(mutex_);

            if ( t->status_ == 1 )
            {
                unprivileged_.erase(t->it_);
                stolen = true;
                t->status_ = 0;
            }
            else
            {
                if ( t->status_ == 2 )
                {
                    ZI_ASSERT(t->then_==nullptr);
                    allocator<callable_t> alloc;
                    t->then_ = alloc.allocate(1);
                    alloc.construct(t->then_,
                                    std::bind(std::forward<Args>(args)...));
                    return;
                }
            }
        }

        if ( stolen )
        {
            t->fn_();
            t->fn_ = nullptr;
        }

        std::bind(std::forward<Args>(args)...)();
    }


    template<typename... Args>
    task_handle schedule_unprivileged(Args&&... args)
    {
        task_handle t = std::allocate_shared<unprivileged_task>
            (allocator<unprivileged_task>(), std::forward<Args>(args)...);
        {
            std::lock_guard<std::mutex> g(mutex_);
            unprivileged_.push_front(t);
            t->it_ = unprivileged_.begin();
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
        }
        return t;
    }

}; // class task_manager


}} // namespace znn::v4

#endif
