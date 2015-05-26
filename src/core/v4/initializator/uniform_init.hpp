#pragma once

#include "initializator.hpp"

namespace znn { namespace v4 {

class uniform_init: public initializator
{
private:
    std::uniform_real_distribution<dboule> dis;

    void do_initialize( dboule* v, size_t n ) noexcept override
    {
        initializator::initialize_with_distribution(dis, v, n);
    }

public:
    uniform_init( dboule low, dboule up )
        : dis(low, up)
    {}

    explicit uniform_init( dboule r = 1 )
        : dis(-r, r)
    {}

}; // class uniform_init

}} // namespace znn::v4
