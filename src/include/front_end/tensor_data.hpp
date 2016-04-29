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

#include "volume_data.hpp"

namespace znn { namespace v4 {

template <typename T>
class tensor_data
{
protected:
    std::vector<volume_data<T>>  data_;


public:
    tensor<T> get_patch( vec3i const & pos ) const
    {
        tensor<T> ret;

        for ( auto& d: data_ )
            ret.push_back(d.get_patch(pos));

        return ret;
    }


public:
    vec3i const & dim() const { return data_[0].dim(); }
    vec3i const & fov() const { return data_[0].fov(); }
    vec3i const & off() const { return data_[0].off(); }

    size_t size() const { return data_.size(); }

    box const & bbox()  const { return data_[0].bbox(); }
    box const & range() const { return data_[0].range(); }


public:
    void set_fov( vec3i const & fov )
    {
        for ( auto& d: data_ ) d.set_fov(fov);
    }

    void set_offset( vec3i const & off )
    {
        for ( auto& d: data_ ) d.set_offset(off);
    }


public:
    void print() const
    {
        std::cout << "[tensor_data]" << "\n"
                  << "dim: " << dim() << "\n"
                  << "fov: " << fov() << "\n"
                  << "off: " << off() << "\n"
                  << "num: " << data_.size() << std::endl;
    }


public:
    explicit
    tensor_data( tensor<T> const & data,
                 vec3i const & fov = vec3i::zero,
                 vec3i const & off = vec3i::zero )
    {
        for ( auto& d: data )
            data_.emplace_back(d,fov,off);
    }

    virtual ~tensor_data() {}

}; // class tensor_data


template <typename T>
class rw_tensor_data: public tensor_data<T>
{
public:
    void set_patch( vec3i const & pos, tensor<T> const & data )
    {
        ZI_ASSERT(this->data_.size()==data.size());

        for ( size_t i = 0; i < data.size(); ++i )
            this->data_[i].set_patch(pos, data[i]);
    }

public:
    explicit
    rw_tensor_data( tensor<T> const & data,
                    vec3i const & fov = vec3i::zero,
                    vec3i const & off = vec3i::zero )
        : tensor_data<T>(data,fov,off)
    {}

    ~rw_tensor_data() {}

}; // class rw_tensor_data

}} // namespace znn::v4