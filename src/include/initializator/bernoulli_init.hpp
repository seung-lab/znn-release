#pragma once

#include "initializator.hpp"

namespace znn { namespace v4 {

template <typename T>
class bernoulli_init: public initializator<T>
{
private:
    std::bernoulli_distribution dis;

    void do_initialize( T* v, size_t n ) noexcept override
    {
        initializator<T>::initialize_with_distribution(dis, v, n);
    }

public:
    explicit bernoulli_init( real p = 0.5 )
        : dis(p)
    {}

}; // class bernoulli_init

template class bernoulli_init<real>;
template class bernoulli_init<bool>;

}} // namespace znn::v4
