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
        return size_[0] * size_[1] * size_[2];
    }

};


}} // namespace znn::v4
