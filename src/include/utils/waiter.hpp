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

    void inc(size_t n = 1)
    {
        std::unique_lock<std::mutex> g(mutex_);
        required_ += n;
    }

    void dec(size_t n = 1)
    {
        std::unique_lock<std::mutex> g(mutex_);
        ZI_ASSERT(current_==0);
        ZI_ASSERT(n<=required_);
        required_ -= n;
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
