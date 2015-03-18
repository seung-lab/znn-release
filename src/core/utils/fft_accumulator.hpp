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

#ifndef ZNN_CORE_UTILS_FFT_ACCUMULATOR_HPP_INCLUDED
#define ZNN_CORE_UTILS_FFT_ACCUMULATOR_HPP_INCLUDED

#include "../types.hpp"
#include "../volume_operators.hpp"
#include "../fft/fftw.hpp"

#include <atomic>

namespace zi { namespace znn {

class fft_accumulator
{
private:
    const vec3i size_;

    size_t              required_;
    std::atomic<size_t> current_ ;

    vol_p<complex>      sum_  ;
    std::mutex          mutex_;

    void do_add(vol_p<complex>&& to_add)
    {
        vol_p<complex> previous_sum;
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

    bool add(vol_p<complex>&& v)
    {
        do_add(std::move(v));
        return ++current_ == required_;
    }

    bool add(const cvol_p<complex>& f, const cvol_p<complex>& w)
    {
        vol_p<complex> previous_sum;
        {
            mutex_guard g(mutex_);
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

    vol_p<double> reset()
    {
        ZI_ASSERT(current_.load()==required_);

        vol_p<double> r = fftw::backward(sum_, size_);
        sum_.reset(); current_ = 0;

        return r;
    }

    double weight() const
    {
        return size_[0] * size_[1] * size_[2];
    }

};


}} // namespace zi::znn

#endif // ZNN_CORE_UTILS_FFT_ACCUMULATOR_HPP_INCLUDED
