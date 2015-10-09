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

#include "fft_accumulator.hpp"
#include "../types.hpp"
#include "../convolution/convolution.hpp"
#include "../cube/cube_operators.hpp"

#include <map>
#include <vector>

namespace znn { namespace v4 {

template<bool Forward>
class accumulator
{
private:
    const vec3i size_;

    map<vec3i,size_t>                        bucket_map_;
    vector<std::unique_ptr<fft_accumulator>> buckets_   ;

    size_t required_;
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
                    return ++current_ == required_;
                }
                previous_sum = std::move(sum_);
            }
            *to_add += *previous_sum;
        }
    }

    bool merge_bucket(size_t b)
    {
        cube_p<real> f = buckets_[b]->reset();
        if ( Forward )
            f = crop_right(*f, size_);
        else
            flip(*f);

        *f /= buckets_[b]->weight();
        return do_add(std::move(f));
    }

public:
    explicit accumulator(const vec3i& size, std::size_t n = 0)
        : size_(size)
        , bucket_map_()
        , buckets_()
        , required_(n)
        , current_(0)
        , sum_()
    {}

    size_t grow(size_t n)
    {
        ZI_ASSERT(current_==0);
        required_ += n;
        return required_;
    }

    size_t grow_fft(const vec3i& size, size_t n)
    {
        ZI_ASSERT(current_==0);

        if ( bucket_map_.count(size) == 0 )
        {
            bucket_map_[size] = buckets_.size();
            buckets_.emplace_back(new fft_accumulator(size,n));
            ++required_;
        }
        else
        {
            buckets_[bucket_map_[size]]->grow(n);
        }
        return bucket_map_[size];
    }

    //
    // sum += f
    //
    bool add(cube_p<real>&& f)
    {
        ZI_ASSERT(current_<required_);
        return do_add(std::move(f));
    }

    //
    // sum += conv(f,w)
    //
    bool add(const ccube_p<real>& f, const ccube_p<real>& w,
             const vec3i& sparse = vec3i::one )
    {
        ZI_ASSERT(current_<required_);

        cube_p<real> previous_sum;

        {
            guard g(mutex_);
            previous_sum = std::move(sum_);
        }

        if ( previous_sum )
        {
            // can convolve_add
            if ( Forward )
            {
                convolve_sparse_add(*f, *w, sparse, *previous_sum);
            }
            else
            {
                convolve_sparse_inverse_add(*f, *w, sparse, *previous_sum);
            }
        }
        else
        {
            // convolve to temp v
            if ( Forward )
            {
                previous_sum = convolve_sparse(*f, *w, sparse);
            }
            else
            {
                previous_sum = convolve_sparse_inverse(*f, *w, sparse);
            }
        }

        return do_add(std::move(previous_sum));
    }

    //
    // buckets[bucket] += f
    //
    bool add(size_t bucket, cube_p<complex>&& f)
    {
        ZI_ASSERT(current_<required_);

        if ( buckets_[bucket]->add(std::move(f)) )
        {
            return merge_bucket(bucket);
        }
        return false;
    }

    //
    // buckets[bucket] += f .* w
    //
    bool add(size_t bucket, const ccube_p<complex>& f, const ccube_p<complex>& w)
    {
        ZI_ASSERT(current_<required_);

        if ( buckets_[bucket]->add(f,w) )
        {
            return merge_bucket(bucket);
        }
        return false;
    }

    cube_p<real> reset()
    {
        ZI_ASSERT(current_==required_);
        current_ = 0;
        return std::move(sum_);
    }

    size_t required() const
    {
        return required_;
    }

};

typedef accumulator<true>  forward_accumulator ;
typedef accumulator<false> backward_accumulator;

}} // namespace znn::v4
