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

#include "dataset.hpp"
#include "tensor_data.hpp"
#include "box.hpp"
#include "../initializator/uniform_init.hpp"

namespace znn { namespace v4 {

template<typename T>
class volume_dataset: public dataset<T>
{
private:
    typedef std::map<std::string, std::pair<vec3i,size_t>>  layers_spec_t;
    typedef std::map<std::string, vec3i>                    sample_spec_t;
    typedef std::unique_ptr<tensor_data<T>>                 tensor_data_p;

private:
    std::map<std::string, tensor_data_p>    data_ ;
    sample_spec_t                           spec_ ;
    box                                     range_;

public:
    box const & range() const { return range_; }


public:
    sample<T> random_sample() override
    {
        ZI_ASSERT(!spec_.empty());
        return get_sample(random_location());
    }

    sample<T> next_sample() override
    {
        ZI_ASSERT(!spec_.empty());
        // TODO(lee): temporary implementation
        return random_sample();
    }

public:
    sample<T> get_sample( vec3i const & loc )
    {
        ZI_ASSERT(!spec_.empty());

        sample<T> ret;

        for ( auto& s: spec_ )
        {
            auto const & name = s.first;
            ZI_ASSERT(data_.count(name)!=0);
            ret[name] = data_[name]->get_patch(loc);
        }

        return ret;
    }

private:
    vec3i random_location() const
    {
        vec3i size = range_.size();

        // thread-safe random number generation
        real p[3];
        uniform_init init(0,1);
        init.initialize(p,3);

        auto z = static_cast<int64_t>(size[0]*p[0]) % size[0];
        auto y = static_cast<int64_t>(size[1]*p[1]) % size[1];
        auto x = static_cast<int64_t>(size[2]*p[2]) % size[2];

        return range_.min() + vec3i(z,y,x);
    }


public:
    void set_spec( layers_spec_t const & spec )
    {
        sample_spec_t ss;
        for ( auto& s: spec )
            ss[s.first] = s.second.first;
        set_spec(ss);
    }

    void set_spec( sample_spec_t const & spec )
    {
        spec_ = spec;
        update_range();
    }

private:
    void update_range()
    {
        range_ = box(); // empty box

        for ( auto& layer: spec_ )
        {
            auto const & name = layer.first;
            auto const & dim  = layer.second;

            ZI_ASSERT(data_.count(name)!=0);
            ZI_ASSERT(data_[name]->size()!=0);

            // update patch size
            data_[name]->set_fov(dim);

            // update valid range
            box const & range = data_[name]->range();
            range_ = range_.empty() ? range : range_.intersect(range);
        }
    }


public:
    // cube
    void add_data( std::string const & name,
                   cube_p<T> const & data,
                   vec3i const & offset = vec3i::zero )
    {
        tensor<T> t = {data};
        add_data(name, t, offset);
    }

    // tensor
    void add_data( std::string const & name,
                   tensor<T> const & data,
                   vec3i const & offset = vec3i::zero )
    {
        data_[name] =
            std::make_unique<tensor_data<T>>(data, vec3i::zero, offset);
    }


public:
             volume_dataset() {}
    virtual ~volume_dataset() {}

}; // class volume_dataset

}} // namespace znn::v4