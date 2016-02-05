#pragma once

#include "global_task_manager.hpp"

#include "malloc.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <functional>
#include <condition_variable>


namespace znn { namespace fwd {

class task_package
{
public:
    task_package(const task_package&) = delete;
    task_package& operator=(const task_package&) = delete;

    task_package(task_package&& other) = delete;
    task_package& operator=(task_package&&) = delete;


private:
    typedef std::function<void(void*)> callable_t;

private:
    std::vector<callable_t>       tasks_;
    std::atomic<int>              size_ ;
    std::atomic<int>              running_threads_;

    std::mutex                    m_ ;
    std::condition_variable       cv_;

    std::size_t                   threads_;

private:
    void loop( void * stack )
    {
        while (1)
        {
            int tid = --size_;
            if ( tid >= 0 )
            {
                tasks_[tid](stack);
            }
            else
            {
                if ( --running_threads_ == 0 )
                {
                    std::lock_guard<std::mutex> g(m_);
                    cv_.notify_one();
                }
                return;
            }
        }
    }

public:
    task_package( std::size_t n, std::size_t t = 8 )
        : tasks_(n)
        , size_(0)
        , running_threads_(0)
        , m_()
        , cv_()
        , threads_(t)
    { }

    template<typename... Args>
    void add_task(Args&&... args)
    {
        tasks_[size_++] = std::bind(std::forward<Args>(args)...,
                                    std::placeholders::_1);
    }

    void execute( std::size_t stack_size = 0 )
    {
        std::size_t n_workers = size_.load();

        while ( n_workers > 3 * threads_ )
        {
            n_workers /= 2;
        }

        if ( n_workers == 0 && size_.load() > 0 )
        {
            n_workers = 1;
        }

        //n_workers = 1;
        if ( n_workers > 0 )
        {
            char * stack = nullptr;

            if ( stack_size > 0 )
            {
                stack = znn_malloc<char>(n_workers*stack_size);
            }

            running_threads_ = static_cast<int>(n_workers);

            for ( std::size_t i = 0; i < n_workers; ++i )
            {
                global_task_manager.schedule(&task_package::loop, this,
                    stack + i * stack_size );
            }

            {
                std::unique_lock<std::mutex> g(m_);
                while ( running_threads_.load() > 0 )
                {
                    cv_.wait(g);
                }
            }

            if ( stack_size > 0 )
            {
                znn_free(stack);
            }

            size_ = 0;
        }
    }

};

}} // namespace znn::fwd
