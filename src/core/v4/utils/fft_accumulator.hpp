#pragma once

#include "../types.hpp"
#include "../cube/cube_operators.hpp"
#include "../fft/fftw.hpp"

#include <atomic>

namespace znn { namespace v4 {

class fft_accumulator
{
private:
    const vec3i size_;

    size_t              required_;
    std::atomic<size_t> current_ ;

    cube_p<complex>     sum_  ;
    std::mutex          mutex_;

    void do_add(cube_p<complex>&& to_add)
    {
        cube_p<complex> previous_sum;
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

public:
    explicit fft_accumulator(const vec3i& size, size_t total = 0)
        : size_(size)
        , required_(total)
        , current_(0)
        , sum_()
    {}

    const vec3i& size() const
    {
        return size_;
    }

    size_t grow(size_t n)
    {
        required_ += n;
        return required_;
    }

    bool add(cube_p<complex>&& v)
    {
        do_add(std::move(v));
        return ++current_ == required_;
    }

    bool add(const ccube_p<complex>& f, const ccube_p<complex>& w)
    {
        cube_p<complex> previous_sum;
        {
            guard g(mutex_);
            previous_sum = std::move(sum_);
        }

        if ( previous_sum )
        {
            mad_to(*f,*w,*previous_sum);
        }
        else
        {
            previous_sum = (*f) * (*w);
        }

        do_add(std::move(previous_sum));
        return ++current_ == required_;
    }

    cube_p<real> reset()
    {
        ZI_ASSERT(current_.load()==required_);

        cube_p<real> r = fftw::backward(std::move(sum_), size_);
        sum_.reset();
        current_ = 0;

        return r;
    }

    real weight() const
    {
        return size_[0] * size_[1] * size_[2];
    }

};


}} // namespace znn::v4
