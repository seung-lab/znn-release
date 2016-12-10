//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
//                    2016  Kisuk Lee           <kisuklee@mit.edu>
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

namespace znn { namespace v4 {

class mult_accumulator
{
private:
    size_t required_;
    size_t current_ ;

    cube_p<real>    mult_ ;
    std::mutex      mutex_;

    bool do_mult(cube_p<real>&& to_mult)
    {
        cube_p<real> previous_mult;
        while (1)
        {
            {
                guard g(mutex_);
                if ( !mult_ )
                {
                    mult_ = std::move(to_mult);
                    return ++current_ == required_;
                }
                previous_mult = std::move(mult_);
            }
            *to_mult *= *previous_mult;
        }
    }

public:
    explicit mult_accumulator(std::size_t n = 0)
        : required_(n)
        , current_(0)
        , mult_()
    {}

    size_t grow(size_t n = 1)
    {
        ZI_ASSERT(current_==0);
        required_ += n;
        return required_;
    }

    size_t shrink(size_t n = 1)
    {
        ZI_ASSERT(current_==0);
        ZI_ASSERT(n<=required_);
        required_ -= n;
        return required_;
    }

    //
    // mult *= f
    //
    bool mult(cube_p<real>&& f)
    {
        ZI_ASSERT(current_<required_);
        return do_mult(std::move(f));
    }

    cube_p<real> reset()
    {
        ZI_ASSERT(current_==required_);
        current_ = 0;
        return std::move(mult_);
    }

    void reset(size_t n)
    {
        ZI_ASSERT(current_=0);
        ZI_ASSERT(!mult_);
        required_ = n;
    }

    size_t required() const
    {
        return required_;
    }

};

}} // namespace znn::v4
