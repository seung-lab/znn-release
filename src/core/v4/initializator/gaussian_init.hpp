#pragma once

#include "initializator.hpp"

namespace znn { namespace v4 {

class gaussian_init: public initializator
{
private:
    std::normal_distribution<double> dis;

    void do_initialize( double* v, size_t n ) noexcept override
    {
        initializator::initialize_with_distribution(dis, v, n);
    }

public:
    explicit gaussian_init( double mu = 0, double sigma = 0.01 )
        : dis(mu, sigma)
    {}

}; // class gaussian_init

}} // namespace znn::v4
