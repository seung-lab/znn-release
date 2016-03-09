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
#include "../assert.hpp"
#include "../cube/cube_operators.hpp"
#include "box.hpp"
#include "utils.hpp"

namespace znn { namespace v4 {

template <typename T>
class volume_data
{
protected:
    cube_p<T>   data_ ;

    vec3i       dim_  ; // volume dimension
    vec3i       fov_  ; // field of view
    vec3i       off_  ; // offset w.r.t. global origin

    box         bbox_ ; // bounding box
    box         range_; // range


public:
    cube_p<T> get_patch( vec3i const & pos ) const
    {
        // validity check
        STRONG_ASSERT(range_.contains(pos));

        // crop & return patch
        box patch = box::centered_box(pos - off_, fov_);
        return crop(*data_, patch.min(), patch.size());
    }


public:
    vec3i const & dim() const { return dim_; }
    vec3i const & fov() const { return fov_; }
    vec3i const & off() const { return off_; }

    box const & bbox()  const { return bbox_ ; }
    box const & range() const { return range_; }


public:
    void set_fov( vec3i const & fov )
    {
        fov_ = fov == vec3i::zero ? dim_ : minimum(fov, dim_);

        set_range();
    }

    void set_offset( vec3i const & off )
    {
        off_ = off;

        set_bbox();
        set_range();
    }

private:
    void set_bbox()
    {
        bbox_ = box(vec3i::zero,dim_) + off_;
    }

    void set_range()
    {
        // top & bottom margins
        vec3i top = fov_/vec3i(2,2,2);
        vec3i btm = fov_ - top - vec3i::one;

        // min & max corners
        vec3i vmin = off_ + top;
        vec3i vmax = off_ + dim_ - btm;

        // valid range
        range_ = box(vmin,vmax);
    }

public:
    void print() const
    {
        std::cout << "[volume_data]" << "\n"
                  << "dim: " << dim_ << "\n"
                  << "fov: " << fov_ << "\n"
                  << "off: " << off_ << "\n";
    }

public:
    explicit
    volume_data( cube_p<T> const & data,
                 vec3i const & fov = vec3i::zero,
                 vec3i const & off = vec3i::zero )
        : data_(data)
        , dim_(size(*data))
        , fov_(fov)
        , off_(off)
    {
        ZI_ASSERT(dim_!=vec3i::zero);

        set_fov(fov);
        set_offset(off);
    }

    virtual ~volume_data() {}

}; // class volume_data


template <typename T>
class rw_volume_data: public volume_data<T>
{
public:
    void set_patch( vec3i const & pos, cube_p<T> const & data )
    {
        if ( data )
        {
            // validity check
            STRONG_ASSERT(this->range_.contains(pos));
            STRONG_ASSERT(size(*data)==this->fov_);

            // set patch
            box patch  = box::centered_box(pos - this->off_, this->fov_);
            vec3i vmin = patch.min();
            vec3i vmax = patch.max();

            (*this->data_)[indices
                [range(vmin[0],vmax[0])]
                [range(vmin[1],vmax[1])]
                [range(vmin[2],vmax[2])]] = (*data);
        }
    }

    cube_p<T> get_data()
    {
        return this->data_;
    }

public:
    explicit
    rw_volume_data( cube_p<T> const & data,
                    vec3i const & fov = vec3i::zero,
                    vec3i const & off = vec3i::zero )
        : volume_data<T>(data,fov,off)
    {}

    ~rw_volume_data() {}

}; // class rw_volume_data

}} // namespace znn::v4
