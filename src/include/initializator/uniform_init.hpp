#pragma once

#include "initializator.hpp"

namespace znn { namespace v4 {

class uniform_init: public initializator
{
private:
    std::uniform_real_distribution<real> dis;

    void do_initialize( real* v, size_t n ) noexcept override
    {
        initializator::initialize_with_distribution(dis, v, n);
    }

public:
    uniform_init( real low, real up )
        : dis(low, up)
    {}

    explicit uniform_init( real r = 1 )
        : dis(-r, r)
    {}

}; // class uniform_init

}} // namespace znn::v4
