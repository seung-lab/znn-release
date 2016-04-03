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

#include "../types.hpp"
#include "../cube/cube_operators.hpp"

#include <map>
#include <vector>

namespace znn { namespace v4 {

class simple_accumulator
{
private:
    size_t required_;
    size_t disabled_;
    size_t current_ ;

    cube_p<real>       sum_;
    std::mutex         mutex_;

    bool do_add(cube_p<real>&& to_add)
    {
        cube_p<real> previous_sum;
        while (1)
        {
            {
                guard g(mutex_);
                if ( !sum_ )
                {
                    sum_ = std::move(to_add);
                    return ++current_ == effectively_required();
                }
                previous_sum = std::move(sum_);
            }
            *to_add += *previous_sum;
        }
    }

public:
    explicit simple_accumulator(std::size_t n = 0)
        : required_(n)
        , disabled_(0)
        , current_(0)
        , sum_()
    {}

    void inc(size_t n = 1)
    {
        ZI_ASSERT(current_==0);
        required_ += n;
    }

    void dec(size_t n = 1)
    {
        ZI_ASSERT(current_==0);
        ZI_ASSERT(n<=required_);
        required_ -= n;
    }

    size_t enable(bool b)
    {
        ZI_ASSERT(current_==0);

        if ( required_ )
        {
            if ( b )
            {
                ZI_ASSERT(effectively_required()<required_);
                --disabled_;
            }
            else
            {
                ZI_ASSERT(0<effectively_required());
                ++disabled_;
            }
        }
        return effectively_required();
    }

    size_t enable_all(bool b)
    {
        disabled_ = b ? 0 : required_;
        return effectively_required();
    }

    //
    // sum += f
    //
    bool add(cube_p<real>&& f)
    {
        ZI_ASSERT(current_<effectively_required());
        return do_add(std::move(f));
    }

    cube_p<real> reset()
    {
        ZI_ASSERT(current_==effectively_required());
        current_ = 0;
        return std::move(sum_);
    }

    size_t required() const
    {
        return required_;
    }

    size_t effectively_required() const
    {
        return required_ - disabled_;
    }

};


}} // namespace znn::v4
