#pragma once

#include <mutex>
#include <condition_variable>

#include "../types.hpp"

namespace znn { namespace v4 {

class waiter
{
private:
    std::size_t             required_ = 0;
    std::size_t             current_  = 0;
    std::mutex              mutex_;
    std::condition_variable cv_   ;

public:
    waiter()
    {
    }

    waiter(size_t how_many)
        : required_(how_many)
    {
    }

    void inc()
    {
        std::unique_lock<std::mutex> g(mutex_);
        ++required_;
    }

    void set(size_t n)
    {
        std::unique_lock<std::mutex> g(mutex_);
        ZI_ASSERT(current_==0);
        required_ = n;
    }


    void one_done()
    {
        std::unique_lock<std::mutex> g(mutex_);
        ++current_;
        if ( current_ == required_ )
        {
            cv_.notify_one();
        }
    }

    void wait()
    {
        std::unique_lock<std::mutex> g(mutex_);
        while ( current_ < required_ )
        {
            cv_.wait(g);
        }
        ZI_ASSERT(current_==required_);
        current_ = 0;
    }

}; // class waiter;

}} // namespace znn::v4
