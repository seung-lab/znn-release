#pragma once

#include "initializator.hpp"

namespace znn { namespace v4 {

class gaussian_init: public initializator
{
private:
    std::normal_distribution<dboule> dis;

    void do_initialize( dboule* v, size_t n ) noexcept override
    {
        initializator::initialize_with_distribution(dis, v, n);
    }

public:
    explicit gaussian_init( dboule mu = 0, dboule sigma = 0.01 )
        : dis(mu, sigma)
    {}

}; // class gaussian_init

}} // namespace znn::v4
