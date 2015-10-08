#pragma once

#include "../types.hpp"
#include "../cube/cube_operators.hpp"

#include <map>
#include <vector>

namespace znn { namespace v4 {

class simple_accumulator
{
private:
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

public:
    explicit simple_accumulator(std::size_t n = 0)
        : required_(n)
        , current_(0)
        , sum_()
    {}

    //
    // sum += f
    //
    bool add(cube_p<real>&& f)
    {
        ZI_ASSERT(current_<required_);
        return do_add(std::move(f));
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


}} // namespace znn::v4
