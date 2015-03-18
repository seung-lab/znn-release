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

#if 0

#ifndef ZNN_CORE_ACCUMULATOR_HPP_INCLUDED
#define ZNN_CORE_ACCUMULATOR_HPP_INCLUDED

#include "types.hpp"
#include "volume_operators.hpp"
#include "volume_utils.hpp"
#include "convolution/convolution.hpp"
#include "fft/fftw.hpp"

#include <mutex>
#include <cstddef>

namespace zi {
namespace znn {

namespace detail {

class fft_accumulator
{
private:
    const vec3i  size_;

    size_t total_  ;
    size_t current_;

    vol_p<complex> value_;
    std::mutex     mutex_;

    bool add(vol_p<complex> val, size_t n)
    {
        vol_p<complex> value;

        while (1)
        {
            {
                mutex_guard g(mutex_);
                if ( current_ == 0 )
                {
                    current_ = n;
                    value_ = std::move(val);
                    return current_ == total_;
                }

                n += current_;
                current_ = 0;
                std::swap(value, value_);
            }

            *val += *value;
        }
    }


public:
    explicit fft_accumulator(const vec3i& size, size_t total = 0)
        : size_(size)
        , total_(total)
        , current_(0)
        , value_()
        , mutex_()
    {}

    const vec3i& size() const
    {
        return size_;
    }

    size_t grow(size_t n)
    {
        mutex_guard g(mutex_);
        ZI_ASSERT(current_ == 0);
        total_ += n;
        return total_;
    }

    bool add(const vol_p<complex>& v)
    {
        return add(v, 1);
    }

    bool add(const cvol_p<complex>& f, const cvol_p<complex>& w)
    {
        vol_p<complex> v;
        size_t         n;

        {
            mutex_guard g(mutex_);
            v.swap(value_);
            n = current_;
            current_ = 0;
        }

        if ( n )
        {
            mad_to(*f,*w,*v);
        }
        else
        {
            v = (*f) * (*w);
        }

        return add(v, n + 1);
    }

    vol_p<double> reset()
    {
        mutex_guard g(mutex_);
        ZI_ASSERT(current_==total_);

        vol_p<double> r = fftw::backward(value_, size_);
        current_ = 0;
        value_.reset();
        return r;
    }

    double weight() const
    {
        return size_[0] * size_[1] * size_[2];
    }
};

} // namespace detail

class accumulator
{
private:
    vec3i size_    ;
    bool  backward_;

    std::map<vec3i,size_t>                                bucket_map_;
    std::vector<std::unique_ptr<detail::fft_accumulator>> buckets_   ;

    size_t current_;
    size_t total_  ;

    vol_p<double> value_;
    std::mutex    mutex_;

    bool add(vol_p<double> val, size_t n)
    {
        vol_p<double> value;

        while (1)
        {
            {
                mutex_guard g(mutex_);
                if ( current_ == 0 )
                {
                    current_ = n;
                    value_ = std::move(val);
                    return current_ == total_;
                }

                n += current_;
                current_ = 0;
                std::swap(value, value_);
            }

            *val += *value;
        }
    }

    bool merge_bucket(size_t b)
    {
        vol_p<double> f = buckets_[b]->reset();

        if ( backward_ )
        {
            flipdims(*f);
        }
        else
        {
            f = volume_utils::crop_right(f, size_);
        }

        *f /= buckets_[b]->weight();
        return add(f,1);
    }

public:
    explicit accumulator(const vec3i& size, std::size_t n = 0,
                         bool backward = false)
        : size_(size)
        , backward_(backward)
        , bucket_map_()
        , buckets_()
        , current_(0)
        , total_(n)
        , value_()
        , mutex_()
    { }

    size_t grow(size_t n)
    {
        mutex_guard g(mutex_);
        ZI_ASSERT(current_ == 0);
        total_ += n;
        return total_;
    }

    size_t grow_fft(const vec3i& size, size_t n)
    {
        mutex_guard g(mutex_);
        ZI_ASSERT(current_ == 0);

        if ( bucket_map_.count(size) == 0 )
        {
            bucket_map_[size] = buckets_.size();
            buckets_.emplace_back(new detail::fft_accumulator(size,n));
        }
        else
        {
            buckets_[bucket_map_[size]]->grow(n);
        }

        return bucket_map_[size];
    }

    bool add(const vol_p<double>& f)
    {
        return add(f,1);
    }

    // adds f convolved with w
    bool add(const cvol_p<double>& f, const cvol_p<double>& w,
             const vec3i& sparse = vec3i::one )
    {
        vol_p<double> v;
        size_t        n;

        {
            mutex_guard g(mutex_);
            v.swap(value_);
            n = current_;
            current_ = 0;
        }

        if ( n )
        {
            // can convolve_add
            if ( backward_ )
            {
                convolve_sparse_inverse_add(*f, *w, sparse, *v);
            }
            else
            {
                convolve_sparse_add(*f, *w, sparse, *v);
            }
        }
        else
        {
            // convolve to temp v
            if ( backward_ )
            {
                v = convolve_sparse_inverse(*f, *w, sparse);
            }
            else
            {
                v = convolve_sparse(*f, *w, sparse);
            }
        }

        return add(v, n + 1);

    }

    bool add_fft(size_t bucket, const vol_p<complex>& f)
    {
        if ( buckets_[bucket]->add(f) )
        {
            return merge_bucket(bucket);
        }
        return false;
    }

    bool add_fft(size_t bucket,
                 const cvol_p<complex>& f,
                 const cvol_p<complex>& w)
    {
        if ( buckets_[bucket]->add(f,w) )
        {
            return merge_bucket(bucket);
        }
        return false;
    }
};


}} // namespace zi::znn

#endif

#endif // ZNN_CORE_ACCUMULATOR_HPP_INCLUDED
