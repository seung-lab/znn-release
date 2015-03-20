#pragma once

#include "initializator.hpp"
#include <algorithm>

namespace znn { namespace v4 {

class constant_init: public initializator
{
private:
    double c_;

    void do_initialize( double*v, size_t n ) noexcept override
    {
        std::fill_n(v, n, c_);
    }

public:
    explicit constant_init( double c = 0 ): c_(c) {}

}; // class constant_init

}} // namespace znn::v4
