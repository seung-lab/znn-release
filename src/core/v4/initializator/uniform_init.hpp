#pragma once

#include "initializator.hpp"

namespace znn { namespace v4 {

class uniform_init: public initializator
{
private:
    std::uniform_real_distribution<double> dis;

    void do_initialize( double* v, size_t n ) noexcept override
    {
        initializator::initialize_with_distribution(dis, v, n);
    }

public:
    uniform_init( double low, double up )
        : dis(low, up)
    {}

    explicit uniform_init( double r = 1 )
        : dis(-r, r)
    {}

}; // class uniform_init

}} // namespace znn::v4
