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

    std::map<vec3i,size_t>                        bucket_map_;
    std::vector<std::unique_ptr<fft_accumulator>> buckets_   ;

    size_t              required_;
    std::atomic<size_t> current_ ;

    cube_p<real>       sum_;
    std::mutex          mutex_;

    void do_add(cube_p<real>&& to_add)
    {
        cube_p<real> previous_sum;
        while (1)
        {
            {
                guard g(mutex_);
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

    bool add(cube_p<real>&& f)
    {
        ZI_ASSERT(current_<required_);
        do_add(std::move(f));
        return ++current_ == required_;
    }

    // adds f convolved with w
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

        do_add(std::move(previous_sum));
        return ++current_ == required_;

    }

    bool add(size_t bucket, cube_p<complex>&& f)
    {
        ZI_ASSERT(current_<required_);

        if ( buckets_[bucket]->add(std::move(f)) )
        {
            merge_bucket(bucket);
            return ++current_ == required_;
        }
        return false;
    }

    bool add(size_t bucket, const ccube_p<complex>& f, const ccube_p<complex>& w)
    {
        ZI_ASSERT(current_<required_);

        if ( buckets_[bucket]->add(f,w) )
        {
            merge_bucket(bucket);
            return ++current_ == required_;
        }
        return false;
    }

    cube_p<real> reset()
    {
        ZI_ASSERT(current_.load()==required_);
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
