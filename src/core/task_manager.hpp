//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZI_ZNN_CORE_TASK_MANAGER_HPP_INCLUDED
#define ZI_ZNN_CORE_TASK_MANAGER_HPP_INCLUDED

#include <functional>
#include <thread>
#include <map>
#include <list>
#include <mutex>
#include <condition_variable>
#include <iostream>

#include <zi/utility/singleton.hpp>

namespace zi {
namespace znn {

template< typename F, typename A >
void trivial_callback_function(const F& f, const A& a)
{
   f(a());
}

class async_thread_pool
{
private:
   std::map<std::size_t, std::list<std::function<void()>>> tasks_;

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
           std::unique_lock<std::mutex> g(mutex_);

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
           std::function<void()> f;

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

               f = std::move(next_task());
           }

           f();
       }
   }

public:
   async_thread_pool()
       : spawned_threads_{0}
       , concurrency_{0}
       , idle_threads_{0}
   {
       set_concurrency(std::thread::hardware_concurrency());
   }

   async_thread_pool(const async_thread_pool&) = delete;
   async_thread_pool& operator=(const async_thread_pool&) = delete;

   async_thread_pool(async_thread_pool&&) = delete;
   async_thread_pool& operator=(async_thread_pool&&) = delete;

   ~async_thread_pool()
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
           std::thread t(&async_thread_pool::worker_loop, this);
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
       std::unique_lock<std::mutex> g(mutex_);
       return concurrency_;
   }

   std::size_t idle_threads()
   {
       std::unique_lock<std::mutex> g(mutex_);
       return idle_threads_;
   }

   std::size_t active_threads()
   {
       std::unique_lock<std::mutex> g(mutex_);
       return concurrency_ - idle_threads_;
   }

private:
   std::function<void()> next_task()
   {
       std::function<void()> f = std::move(tasks_.rbegin()->second.front());

       tasks_.rbegin()->second.pop_front();
       if ( tasks_.rbegin()->second.size() == 0 )
       {
           tasks_.erase(tasks_.rbegin()->first);
       }
       return f;
   }

public:
   void add_task(std::size_t priority, std::function<void()>&& f)
   {
       std::unique_lock<std::mutex> g(mutex_);
       tasks_[priority].emplace_back(std::forward<std::function<void()>>(f));
       if ( idle_threads_ > 0 ) workers_cv_.notify_all();
   }

}; // class async_thread_pool

namespace {
async_thread_pool& async_thread_pool_instance =
   singleton<async_thread_pool>::instance();
}

template<typename... Args>
void async(Args&&... args)
{
   async_thread_pool_instance.add_task
       (0, std::bind(std::forward<Args>(args)...));
}

template<typename... Args>
void async_priority(std::size_t priority, Args&&... args)
{
   async_thread_pool_instance.add_task
       (priority, std::bind(std::forward<Args>(args)...));
}

template<typename F, typename... Args>
void async_cb(F&& f, Args&&... args)
{
   async_thread_pool_instance.add_task
       (0, std::bind(std::forward<F>(f),
                     std::bind(std::forward<Args>(args)...)));
}

template<typename F, typename... Args>
void async_priority_cb(F&& f, std::size_t priority, Args&&... args)
{
   async_thread_pool_instance.add_task
       (priority, std::bind(std::forward<F>(f),
                            std::bind(std::forward<Args>(args)...)));
}


std::size_t get_concurrency()
{
   return async_thread_pool_instance.get_concurrency();
}

std::size_t set_concurrency(std::size_t n)
{
   return async_thread_pool_instance.set_concurrency(n);
}


} // namespace zi::znn

}



#endif //ZI_ZNN_CORE_TASK_MANAGER_HPP_INCLUDED
