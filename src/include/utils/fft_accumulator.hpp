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

    size_t required_;
    size_t current_ ;

    cube_p<complex>     sum_  ;
    std::mutex          mutex_;

    std::unique_ptr<fftw::transformer> fftw_;

    real weight_ = 0;

    bool do_add(cube_p<complex>&& to_add)
    {
        cube_p<complex> previous_sum;
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

public:
    explicit fft_accumulator(const vec3i& size, size_t total = 0)
        : size_(size)
        , required_(total)
        , current_(0)
        , sum_()
        , fftw_(std::make_unique<fftw::transformer>(size))
    {
        vec3i s = fftw_->actual_size();
        weight_ = s[0] * s[1] * s[2];
    }

    fftw::transformer const & get_transformer() const
    {
        return *fftw_;
    }

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
        return do_add(std::move(v));
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

        return do_add(std::move(previous_sum));
    }

    cube_p<real> reset()
    {
        ZI_ASSERT(current_==required_);

        cube_p<real> r = fftw_->backward(std::move(sum_));
        sum_.reset();
        current_ = 0;

        return r;
    }

    real weight() const
    {
        return weight_;
    }

};


}} // namespace znn::v4
