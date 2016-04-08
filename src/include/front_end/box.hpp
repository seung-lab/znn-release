//
// Copyright (C)      2016  Kisuk Lee           <kisuklee@mit.edu>
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
#include "utils.hpp"

namespace znn { namespace v4 {

class box
{
private:
    vec3i   vmin_;  // min corner of the box
    vec3i   vmax_;  // max corner of the box
    vec3i   size_;  // size of the box


public:
    vec3i const & min()  const { return vmin_; }
    vec3i const & max()  const { return vmax_; }
    vec3i const & size() const { return size_; }

    bool empty() const
    {
        return size_ == vec3i::zero;
    }

    bool contains( vec3i const & p ) const
    {
        return (vmin_[0] <= p[0]) && (p[0] < vmax_[0]) &&
               (vmin_[1] <= p[1]) && (p[1] < vmax_[1]) &&
               (vmin_[2] <= p[2]) && (p[2] < vmax_[2]);
    }

    bool contains( box const & rhs ) const
    {
        return intersect(rhs) == rhs;
    }

    bool overlaps( box const & rhs ) const
    {
        return (vmax_[0] > rhs.vmin_[0]) && (vmin_[0] < rhs.vmax_[0]) &&
               (vmax_[1] > rhs.vmin_[1]) && (vmin_[1] < rhs.vmax_[1]) &&
               (vmax_[2] > rhs.vmin_[2]) && (vmin_[2] < rhs.vmax_[2]);
    }

    static
    box intersect( box const & a, box const & b )
    {
        box r;

        if ( a.overlaps(b) )
        {
            vec3i vmin = maximum(a.min(),b.min());
            vec3i vmax = minimum(a.max(),b.max());

            r = box(vmin,vmax);
        }

        return r;
    }

    box intersect( box const & rhs ) const
    {
        return intersect(*this, rhs);
    }

    static
    box merge( box const & a, box const & b )
    {
        vec3i vmin = minimum(a.min(),b.min());
        vec3i vmax = maximum(a.max(),b.max());

        return box(vmin,vmax);
    }

    box merge( box const & rhs ) const
    {
        return merge(*this, rhs);
    }

    void translate( vec3i const & offset )
    {
        vmin_ += offset;
        vmax_ += offset;
    }


public:
    bool operator==( box const & rhs )
    {
        return (vmin_ == rhs.vmin_) && (vmax_ == rhs.vmax_);
    }

    bool operator!=( box const & rhs )
    {
        return !(*this == rhs);
    }

    box operator+( vec3i const & offset )
    {
        return box(vmin_ + offset, vmax_ + offset);
    }

    box operator-( vec3i const & offset )
    {
        return box(vmin_ - offset, vmax_ - offset);
    }

    box operator+( box const & rhs )
    {
        return this->merge(rhs);
    }

public:
    friend std::ostream&
    operator<<( std::ostream & os, box const & rhs )
    {
        return (os << "box(" << rhs.size_ << ")"
                   << "[" << rhs.vmin_ << "][" << rhs.vmax_ << "]");
    }


public:
    static
    box centered_box( vec3i const & center, vec3i const & size )
    {
        vec3i rad = size/vec3i(2,2,2);

        vec3i vmin = center - rad;
        vec3i vmax = vmin + size;

        return box(vmin,vmax);
    }


public:
    explicit box( vec3i const & v1 = vec3i::zero,
                  vec3i const & v2 = vec3i::zero )
        : vmin_(minimum(v1,v2))
        , vmax_(maximum(v1,v2))
        , size_(vmax_ - vmin_)
    {}

    ~box() {}

}; // class box

}} // namespace znn::v4