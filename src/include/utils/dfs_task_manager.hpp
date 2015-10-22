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
#include <boost/lockfree/stack.hpp>

#include "log.hpp"
#include "global_task_manager.hpp"

namespace znn { namespace v4 {

namespace {
thread_local size_t znn_thread_id = 0;
}

struct regular_task
{
    std::function<void()> fn;
    size_t                thread_id;
    list<regular_task*>::const_iterator local ;
    list<regular_task*>::const_iterator global;

    template<class... Args>
    explicit regular_task( size_t tid,
                           Args && ... args )
        : thread_id(tid)
        , fn(std::bind(std::forward<Args>(args)...))
    {}
};

class dfs_task_manager
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

        friend class dfs_task_manager;

    public:
        template<class... Args>
        explicit unprivileged_task(Args&&... args)
            : fn_(std::bind(std::forward<Args>(args)...))
        {}
    };

public:
    typedef std::shared_ptr<unprivileged_task> task_handle;

private:
    std::size_t spawned_threads_;
    std::size_t concurrency_    ;
    std::size_t idle_threads_   ;

    std::mutex              mutex_;
    std::condition_variable manager_cv_;
    std::condition_variable workers_cv_;

    list<task_handle>                   unprivileged_  ;
    list<regular_task*>                 tasks_         ;
    list<regular_task*>                 local_tasks_[1000];
    size_t                              tot_tasks_ = 0 ;

private:
    void worker_loop()
    {
        {
            std::lock_guard<std::mutex> g(mutex_);

            if ( spawned_threads_ >= concurrency_ )
            {
                return;
            }

            znn_thread_id = ++spawned_threads_;
            if ( spawned_threads_ == concurrency_ )
            {
                manager_cv_.notify_all();
            }
        }

        task_handle f2;

        while (true)
        {
            regular_task* f1 = nullptr;

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
                f1->fn();
                allocator<regular_task> alloc;
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
    dfs_task_manager(std::size_t concurrency = std::thread::hardware_concurrency())
        : spawned_threads_{0}
        , concurrency_{0}
        , idle_threads_{0}
    {
        set_concurrency(concurrency);
    }

    dfs_task_manager(const dfs_task_manager&) = delete;
    dfs_task_manager& operator=(const dfs_task_manager&) = delete;

    dfs_task_manager(dfs_task_manager&& other) = delete;
    dfs_task_manager& operator=(dfs_task_manager&&) = delete;

    ~dfs_task_manager()
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
            //std::thread t(&dfs_task_manager::worker_loop, this);
            //t.detach();
            global_task_manager.schedule(&dfs_task_manager::worker_loop, this);
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
    regular_task* next_task()
    {
        regular_task* f;
        size_t const id = znn_thread_id;

        if ( local_tasks_[id].size() )
        {
            f = local_tasks_[id].front();
            local_tasks_[id].erase(f->local);
            tasks_.erase(f->global);
        }
        else
        {
            f = tasks_.front();
            local_tasks_[f->thread_id].erase(f->local);
            tasks_.erase(f->global);
        }

        --tot_tasks_;

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
    void schedule(std::size_t, Args&&... args)
    {
        size_t const id = znn_thread_id;

        allocator<regular_task> alloc;
        regular_task* t = alloc.allocate(1);

        alloc.construct(t, id, std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            t->local  = local_tasks_[id].insert(local_tasks_[id].begin(),t);
            t->global = tasks_.insert(tasks_.begin(),t);
            ++tot_tasks_;
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
        }
    }

    template<typename... Args>
    void asap(Args&&... args)
    {
        schedule(0, std::forward<Args>(args)...);
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

        allocator<callable_t> alloc;
        callable_t* after = alloc.allocate(1);
        alloc.construct(after, std::bind(std::forward<Args>(args)...));

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
                    t->then_ = after;
                    return;
                }
            }
        }

        if ( stolen )
        {
            t->fn_();
            t->fn_ = nullptr;
        }

        (*after)();
        alloc.destroy(after);
        alloc.deallocate(after,1);
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

}; // class dfs_task_manager

using task_manager = dfs_task_manager;

}} // namespace znn::v4
