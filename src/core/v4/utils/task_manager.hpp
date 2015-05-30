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

#ifndef ZNN_ANALYSE_TASK_MANAGER

class task_manager
{
private:
    typedef std::function<void()> callable_t;

    size_t next_unprivileged_task_id;

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
    typedef size_t task_handle;

private:
    std::map<std::size_t, std::list<callable_t*>> tasks_         ;
    size_t                                        tot_tasks_ = 0 ;
    std::map<std::size_t, unprivileged_task*>     unprivileged_  ;

    // executing in one of the task manager's threads!
    std::map<std::size_t, unprivileged_task*>     executing_   ;

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

        std::pair<size_t, unprivileged_task*> f2;

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
    void execute_unprivileged_task
    (std::pair<size_t, unprivileged_task*> const & t)
    {
        t.second->fn_();
        callable_t* after = nullptr;

        {
            std::unique_lock<std::mutex> g(mutex_);
            after    = t.second->then_;
            executing_.erase(t.first);
        }

        if ( after )
        {
            (*after)();
            delete after;
        }

        delete t.second;
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

        --tot_tasks_;
        // LOG(info) << tot_tasks_
        //           << ", " << unprivileged_.size() << ";";

        return f;
    }

    std::pair<const size_t, unprivileged_task*> next_unprivileged_task()
    {
        auto x = *unprivileged_.rbegin();
        unprivileged_.erase(x.first);
        executing_.insert(x);

        // LOG(info) << tot_tasks_
        //           << ", " << unprivileged_.size() << ";";

        return x;
    }

public:
    template<typename... Args>
    void schedule(std::size_t priority, Args&&... args)
    {
        callable_t* fn = new callable_t(std::bind(std::forward<Args>(args)...));
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
        callable_t* fn = new callable_t(std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            tasks_[std::numeric_limits<std::size_t>::max()].emplace_front(fn);
            ++tot_tasks_;
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
        }
    }

    template<typename... Args>
    void require_done(size_t t, Args&&... args)
    {
        // doesn't exist!
        if ( t == 0 )
        {
            std::bind(std::forward<Args>(args)...)();
            return;
        }

        unprivileged_task* stolen = nullptr;

        {
            std::lock_guard<std::mutex> g(mutex_);
            if ( unprivileged_.count(t) )
            {
                stolen = unprivileged_[t];
                unprivileged_.erase(t);
            }
            else
            {
                if ( executing_.count(t) )
                {
                    ZI_ASSERT(executing_[t]->then_==nullptr);
                    executing_[t]->then_ =
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
            unprivileged_[++next_unprivileged_task_id] = t;
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
            return next_unprivileged_task_id;
        }
    }

}; // class task_manager

#else


class task_manager
{
private:
    typedef std::function<void()> callable_t;

    size_t next_unprivileged_task_id;

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
    typedef size_t task_handle;

private:
    std::map<std::size_t, std::list<callable_t*>> tasks_         ;
    size_t                                        tot_tasks_ = 0 ;
    std::map<std::size_t, unprivileged_task*>     unprivileged_  ;

    // executing in one of the task manager's threads!
    std::map<std::size_t, unprivileged_task*>     executing_   ;

    std::size_t spawned_threads_;
    std::size_t concurrency_    ;
    std::size_t idle_threads_   ;

    std::mutex              mutex_;
    std::condition_variable manager_cv_;
    std::condition_variable workers_cv_;

    struct timepoint
    {
        float time;
        float priv;
        float unpriv;
        float idle;
    };

    mutable std::vector<timepoint> timepoints_;
    mutable zi::wall_timer wt_;

    void stamp()
    {
        float time = static_cast<float>(wt_.elapsed<double>());
        float priv = 0;
        if ( tasks_.size() )
            priv = static_cast<float>(tasks_.rbegin()->second.size());
        float unpr = static_cast<float>(unprivileged_.size());
        float idle = static_cast<float>(idle_threads_);
        timepoints_.push_back({time,priv,unpr,idle});
    }

public:
    void dump()
    {
        std::lock_guard<std::mutex> g(mutex_);

        std::ofstream fvol("/tmp/dat.dat", (std::ios::out | std::ios::binary) );
        fvol.write( reinterpret_cast<char*>(&(timepoints_[0])), timepoints_.size() * sizeof(timepoint) );

        // std::cout << '[';
        // for ( auto& d: timepoints_ )
        // {
        //     std::cout << d.time << ',' << d.priv << ','
        //               << d.unpriv << ',' << d.idle << ';' << std::endl;
        // }
        // std::cout << ']';
    }

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

        std::pair<size_t, unprivileged_task*> f2;

        while (true)
        {
            callable_t* f1 = nullptr;

            {
                std::unique_lock<std::mutex> g(mutex_);

                while ( tasks_.empty() &&
                        unprivileged_.empty() &&
                        concurrency_ >= spawned_threads_ )
                {
                    stamp();
                    ++idle_threads_;
                    stamp();
                    workers_cv_.wait(g);
                    stamp();
                    --idle_threads_;
                    stamp();
                }
                stamp();


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
    void execute_unprivileged_task
    (std::pair<size_t, unprivileged_task*> const & t)
    {
        t.second->fn_();
        callable_t* after = nullptr;

        {
            std::unique_lock<std::mutex> g(mutex_);
            after    = t.second->then_;
            executing_.erase(t.first);
        }

        if ( after )
        {
            (*after)();
            delete after;
        }

        delete t.second;
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

        --tot_tasks_;
        // LOG(info) << tot_tasks_
        //           << ", " << unprivileged_.size() << ";";

        return f;
    }

    std::pair<const size_t, unprivileged_task*> next_unprivileged_task()
    {
        auto x = *unprivileged_.rbegin();
        unprivileged_.erase(x.first);
        executing_.insert(x);

        // LOG(info) << tot_tasks_
        //           << ", " << unprivileged_.size() << ";";

        return x;
    }

public:
    template<typename... Args>
    void schedule(std::size_t priority, Args&&... args)
    {
        callable_t* fn = new callable_t(std::bind(std::forward<Args>(args)...));
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
        callable_t* fn = new callable_t(std::bind(std::forward<Args>(args)...));
        {
            std::lock_guard<std::mutex> g(mutex_);
            tasks_[std::numeric_limits<std::size_t>::max()].emplace_front(fn);
            ++tot_tasks_;
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
        }
    }

    template<typename... Args>
    void require_done(size_t t, Args&&... args)
    {
        // doesn't exist!
        if ( t == 0 )
        {
            std::bind(std::forward<Args>(args)...)();
            return;
        }

        unprivileged_task* stolen = nullptr;

        {
            std::lock_guard<std::mutex> g(mutex_);
            if ( unprivileged_.count(t) )
            {
                stolen = unprivileged_[t];
                unprivileged_.erase(t);
            }
            else
            {
                if ( executing_.count(t) )
                {
                    ZI_ASSERT(executing_[t]->then_==nullptr);
                    executing_[t]->then_ =
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
            unprivileged_[++next_unprivileged_task_id] = t;
            if ( idle_threads_ > 0 ) workers_cv_.notify_one();
            return next_unprivileged_task_id;
        }
    }

}; // class task_manager


#endif // ZNN_ANALYSE_TASK_MANAGER

}} // namespace znn::v4
