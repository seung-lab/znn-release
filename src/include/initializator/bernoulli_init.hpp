#pragma once

#include "initializator.hpp"

namespace znn { namespace v4 {

class bernoulli_init: public initializator
{
private:
    std::bernoulli_distribution dis;

    void do_initialize( real* v, size_t n ) noexcept override
    {
        initializator::initialize_with_distribution(dis, v, n);
    }

    void do_initialize( bool* v, size_t n ) noexcept
    {
        guard g(rng);

        for ( size_t i = 0; i < n; ++i )
        {
            v[i] = dis(rng.rng);
        }
    }

public:
    void initialize( bool* v, size_t n ) noexcept
    { this->do_initialize(v,n); }

    void initialize( cube<bool>& v ) noexcept
    {
        this->do_initialize(v.data(), v.num_elements());
    }

    void initialize( const cube_p<bool>& v ) noexcept
    {
        this->do_initialize(v->data(), v->num_elements());
    }

public:
    explicit bernoulli_init( real p = 0.5 )
        : dis(p)
    {}

}; // class bernoulli_init

}} // namespace znn::v4
