//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
//                    2015  Kisuk Lee           <kisuklee@mit.edu>
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

class max_accumulator
{
private:
    size_t required_;
    size_t disabled_;
    size_t current_ ;

    cube_p<real>    maximum_;
    cube_p<int>     indices_;
    std::mutex      mutex_;

    bool do_max(cube_p<real>&& to_add, int idx)
    {
        cube_p<real> previous_max;
        cube_p<int>  previous_idx;
        while (1)
        {
            {
                guard g(mutex_);
                if ( !maximum_ )
                {
                    maximum_ = std::move(to_add);

                    ZI_ASSERT(!indices_);
                    if ( previous_idx )
                    {
                        indices_ = std::move(previous_idx);
                    }
                    else
                    {
                        indices_ = get_cube<int>(size(*maximum_));
                        fill(*indices_,idx);
                    }

                    return ++current_ == required_;
                }
                previous_max = std::move(maximum_);
                previous_idx = std::move(indices_);
            }
            maximum(*previous_max,*previous_idx,*to_add,idx);
        }
    }

    // v  = maximum(c,v)
    // vi = update_indices(vi,idx)
    void maximum( cube<real> const & c, cube<int> & vi,
                  cube<real> & v, int idx )
    {
        ZI_ASSERT(c.num_elements()==v.num_elements());
        const real* src = c.data();
        real*       dest = v.data();
        int*        indices = vi.data();

        for ( size_t i = 0; i < v.num_elements(); ++i )
        {
            if ( src[i] > dest[i] )
            {
                dest[i] = src[i];
            }
            else
            {
                indices[i] = idx;
            }
        }
    }

public:
    explicit max_accumulator(std::size_t n = 0)
        : required_(n)
        , disabled_(0)
        , current_(0)
        , maximum_()
        , indices_()
    {}

    size_t grow(size_t n)
    {
        ZI_ASSERT(current_==0);
        required_ += n;
        return required_;
    }

    size_t disable(size_t n)
    {
        ZI_ASSERT(n<=effectively_required());
        disabled_ += n;
        return effectively_required();
    }

    void enable_all(bool b)
    {
        disabled_ = b ? 0 : required_;
    }

    //
    // maximum = max(maximum, f)
    //
    bool add(cube_p<real>&& f, int idx)
    {
        ZI_ASSERT(current_<required_);
        return do_max(std::move(f),idx);
    }

    std::pair<cube_p<real>,cube_p<int>> reset()
    {
        ZI_ASSERT(current_==required_);
        current_ = 0;
        return { std::move(maximum_), std::move(indices_) };
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
