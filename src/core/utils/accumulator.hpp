//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_UTILS_ACCUMULATOR_HPP_INCLUDED
#define ZNN_CORE_UTILS_ACCUMULATOR_HPP_INCLUDED

#include "fft_accumulator.hpp"
#include "../convolution/convolution.hpp"
#include "../volume_utils.hpp"

#include <map>
#include <vector>

namespace zi { namespace znn {

template<bool Forward>
class accumulator
{
private:
    const vec3i size_;

    std::map<vec3i,size_t>                        bucket_map_;
    std::vector<std::unique_ptr<fft_accumulator>> buckets_   ;

    size_t              required_;
    std::atomic<size_t> current_ ;

    vol_p<double>       sum_;
    std::mutex          mutex_;

    void do_add(vol_p<double>&& to_add)
    {
        vol_p<double> previous_sum;
        while (1)
        {
            {
                mutex_guard g(mutex_);
                if ( !sum_ )
                {
                    sum_ = std::move(to_add);
                    return;
                }
                previous_sum = std::move(sum_);
            }

            *to_add += *previous_sum;
        }
    }

    void merge_bucket(size_t b)
    {
        vol_p<double> f = buckets_[b]->reset();
        if ( Forward )
            f = volume_utils::crop_right(f, size_);
        else
            flipdims(*f);

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
        ZI_ASSERT(current_.load() == 0);
        required_ += n;
        return required_;
    }

    size_t grow_fft(const vec3i& size, size_t n)
    {
        ZI_ASSERT(current_.load() == 0);

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

    bool add(vol_p<double>&& f)
    {
        do_add(std::move(f));
        return ++current_ == required_;
    }

    // adds f convolved with w
    bool add(const cvol_p<double>& f, const cvol_p<double>& w,
             const vec3i& sparse = vec3i::one )
    {
        vol_p<double> previous_sum;

        {
            mutex_guard g(mutex_);
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

        do_add(std::move(previous_sum));
        return ++current_ == required_;

    }

    bool add(size_t bucket, vol_p<complex>&& f)
    {
        if ( buckets_[bucket]->add(std::move(f)) )
        {
            merge_bucket(bucket);
            return ++current_ == required_;
        }
        return false;
    }

    bool add(size_t bucket, const cvol_p<complex>& f, const cvol_p<complex>& w)
    {
        if ( buckets_[bucket]->add(f,w) )
        {
            merge_bucket(bucket);
            return ++current_ == required_;
        }
        return false;
    }

    vol_p<double> reset()
    {
        ZI_ASSERT(current_.load()==required_);
        current_ = 0;
        return std::move(sum_);
    }


};

typedef accumulator<true>  forward_accumulator ;
typedef accumulator<false> backward_accumulator;

}} // namespace zi::znn

#endif // ZNN_CORE_UTILS_ACCUMULATOR_HPP_INCLUDED
