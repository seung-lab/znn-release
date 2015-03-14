//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_CORE_ACCUMULATOR_HPP_INCLUDED
#define ZNN_CORE_ACCUMULATOR_HPP_INCLUDED

#include <mutex>
#include <cstddef>

#include <unistd.h>

namespace zi {
namespace znn {

template<typename T>
class accumulator
{
private:
    std::size_t count_  = 0;
    T           value_  = T();
    std::mutex  mutex_;

public:
    std::size_t add(T val)
    {
        std::size_t count = 1;
        T           value;

        while (1)
        {
            {
                std::unique_lock<std::mutex> g(mutex_);
                if ( count_ == 0 )
                {
                    count_ = count;
                    value_ = val;
                    return count_;
                }

                count += count_;
                count_ = 0;
                std::swap(value, value_);
            }

            usleep(1111);
            val += value;
        }
    }

    T get()
    {
        std::unique_lock<std::mutex> g(mutex_);
        return value_;
    }

    T get_count()
    {
        std::unique_lock<std::mutex> g(mutex_);
        return count_;
    }

    void reset()
    {
        std::unique_lock<std::mutex> g(mutex_);
        value_ = T();
        count_ = 0;
    }

};

}} // namespace zi::znn

#endif // ZNN_CORE_ACCUMULATOR_HPP_INCLUDED
