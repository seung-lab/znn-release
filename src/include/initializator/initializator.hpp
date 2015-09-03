#pragma once

#include <random>
#include <limits>
#include <mutex>
#include <ctime>
#include <zi/utility/singleton.hpp>
#include "../types.hpp"

namespace znn { namespace v4 {

namespace detail {

struct random_number_generator_impl: std::mutex
{
    std::mt19937 rng = std::mt19937(1234);
    // std::mt19937 rng = std::mt19937(std::random_device(0));
}; // class random_number_generator_impl

} // namespace detail

template <typename T>
class initializator
{
protected:
    template <typename D>
    static void initialize_with_distribution(D&& dis, T* v, size_t n) noexcept
    {
        detail::random_number_generator_impl& rng = 
            zi::singleton<detail::random_number_generator_impl>::instance();

        guard g(rng);

        for ( size_t i = 0; i < n; ++i )
        {
            v[i] = dis(rng.rng);
        }
    }

    template <typename D>
    static void initialize_with_distribution(D&& dis, cube<real>& v) noexcept
    {
        initialize_with_distribution(std::forward<T>(dis),
                                     v.data(), v.num_elements());
    }


    virtual void do_initialize( T*, size_t ) noexcept = 0;

public:
    virtual ~initializator() {}

    void initialize( T* v, size_t n ) noexcept
    { this->do_initialize(v,n); }

    void initialize( cube<T>& v ) noexcept
    {
        this->do_initialize(v.data(), v.num_elements());
    }

    void initialize( const cube_p<T>& v ) noexcept
    {
        this->do_initialize(v->data(), v->num_elements());
    }

}; // class initializator

}} // namespace znn::v4

#include "initializators.hpp"
