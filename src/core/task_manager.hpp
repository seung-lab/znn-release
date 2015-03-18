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

#ifndef ZNN_CORE_TASK_MANAGER_HPP_INCLUDED
#define ZNN_CORE_TASK_MANAGER_HPP_INCLUDED

#include <functional>
#include <thread>
#include <atomic>
#include <map>
#include <set>
#include <list>
#include <mutex>
#include <memory>
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
    struct unprivileged_task: std::enable_shared_from_this<unprivileged_task>
    {
    private:
        std::function<void()>  fn_  ;
        callable_t*            then_ = nullptr;

        friend class task_manager;

    public:
        template<class... Args>
        explicit unprivileged_task(Args&&... args)
            : fn_(std::bind(std::forward<Args>(args)...))
        {}
    };

public:
    typedef unprivileged_task* task_handle;

private:
    std::map<std::size_t, std::list<callable_t*>> tasks_       ;
    std::set<unprivileged_task*>                  unprivileged_;

    // executing in one of the task manager's threads!
    std::set<unprivileged_task*>                  executing_   ;

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
            callable_t* f1         = nullptr;
            unprivileged_task* f2  = nullptr;

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
                delete f1;
            }
            else
            {
                execute_unprivileged_task(f2);
            }
        }
    }

private:
    // executing in one of the manaer's threads
    void execute_unprivileged_task(unprivileged_task* t)
    {
        t->fn_();
        callable_t* after = nullptr;

        {
            std::unique_lock<std::mutex> g(mutex_);
            after    = t->then_;
            executing_.erase(t);
        }

        if ( after )
        {
            (*after)();
            delete after;
        }

        delete t;
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

    unprivileged_task* next_unprivileged_task()
    {
        auto x = unprivileged_.cbegin();
        unprivileged_.erase(x);
        return *x;
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
    void asap(Args&&... args)
    {
        callable_t* fn = new callable_t(std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            tasks_[std::numeric_limits<std::size_t>::max()].emplace_front(fn);
            if ( idle_threads_ > 0 ) workers_cv_.notify_all();
        }
    }

    template<typename... Args>
    void require_done(unprivileged_task* t, Args&&... args)
    {
        if ( t == nullptr )
        {
            std::bind(std::forward<Args>(args)...)();
            return;
        }

        unprivileged_task* stolen = nullptr;

        {
            std::lock_guard<std::mutex> g(mutex_);
            if ( unprivileged_.erase(t) )
            {
                stolen = t;
            }
            else
            {
                if ( executing_.count(t) )
                {
                    ZI_ASSERT(t->then_==nullptr);
                    t->then_ =
                        new callable_t(std::bind(std::forward<Args>(args)...));
                    return;
                }
            }
        }

        if ( stolen )
        {
            stolen->fn_();
            delete stolen;
        }

        std::bind(std::forward<Args>(args)...)();
    }


    template<typename... Args>
    task_handle schedule_unprivileged(Args&&... args)
    {
        unprivileged_task* t =
            new unprivileged_task(std::forward<Args>(args)...);
        {
            std::lock_guard<std::mutex> g(mutex_);
            unprivileged_.insert(t);
            if ( idle_threads_ > 0 ) workers_cv_.notify_all();
        }

        return t;
    }

}; // class task_manager


}} // namespace zi::znn


#endif // ZNN_CORE_TASK_MANAGER_HPP_INCLUDED
