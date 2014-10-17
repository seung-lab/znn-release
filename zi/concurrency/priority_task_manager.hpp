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

#ifndef ZI_CONCURRENCY_PRIORITY_TASK_MANAGER_HPP
#define ZI_CONCURRENCY_PRIORITY_TASK_MANAGER_HPP 1

#include <zi/concurrency/detail/priority_task_manager_impl.hpp>
#include <zi/concurrency/detail/priority_map_task_container.hpp>

#include <zi/bits/type_traits.hpp>
#include <zi/meta/enable_if.hpp>

namespace zi {
namespace concurrency_ {

template< class TaskContainer >
class priority_task_manager_tpl
{
private:
    typedef priority_task_manager_impl< TaskContainer > priority_task_manager_t;
    shared_ptr< priority_task_manager_t > manager_;

public:
    priority_task_manager_tpl( std::size_t worker_limit,
                      std::size_t max_size = std::numeric_limits< std::size_t >::max() ) :
        manager_( new priority_task_manager_t( worker_limit, max_size ) )
    {
    }


    std::size_t empty()
    {
        return manager_->empty();
    }

    std::size_t idle()
    {
        return manager_->empty();
    }

    std::size_t size()
    {
        return manager_->size();
    }

    std::size_t worker_count()
    {
        return manager_->worker_count();
    }

    std::size_t worker_limit()
    {
        return manager_->worker_limit();
    }

    std::size_t idle_workers()
    {
        return manager_->idle_workers();
    }

    bool start()
    {
        return manager_->start();
    }

    void stop( bool and_join = false )
    {
        return manager_->stop( and_join );
    }

    void join()
    {
        manager_->join();
    }

    void insert( shared_ptr< runnable > task, std::size_t prio = 0 )
    {
        manager_->insert( task, prio );
    }


    template< class Runnable >
    void insert( shared_ptr< Runnable > task,
                 std::size_t prio = 0,
                 typename meta::enable_if<
                 typename is_base_of< runnable, Runnable >::type >::type* = 0 )
    {
        manager_->insert( task, prio );
    }

    template< class Function >
    void insert( const Function& task,
                 std::size_t prio = 0,
                 typename meta::enable_if<
                 typename is_convertible< Function, function< void() >
                 >::type >::type* = 0 )
    {
        this->insert( shared_ptr< runnable_function_wrapper >
                      ( new runnable_function_wrapper( task ) ), prio);
    }


    void clear()
    {
        manager_->clear();
    }

    void add_workers( std::size_t count )
    {
        manager_->add_workers( count );
    }

    void remove_workers( std::size_t count )
    {
        manager_->remove_workers( count );
    }

};



} // namespace concurrency_

namespace task_manager {

typedef concurrency_::priority_task_manager_tpl<concurrency_::detail::priority_map_task_container>   prioritized;
    ;

} // namespace task_manager

typedef task_manager::prioritized   prioritized_task_manager;
typedef task_manager::prioritized   priority_task_manager   ;


} // namespace zi

#endif


