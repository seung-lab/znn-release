
#pragma once

#include <random>
#include <limits>
#include <mutex>
#include <ctime>
#include <zi/utility/singleton.hpp>
#include "types.hpp"

namespace znn { namespace fwd {

namespace detail {

struct random_number_generator_impl: std::mutex
{
    std::mt19937 rng = std::mt19937(1234);
    // std::mt19937 rng = std::mt19937(std::random_device(0));
}; // class random_number_generator_impl

} // namespace detail

class uniform_init
{
protected:
    template <typename D>
    static void initialize_with_distribution(D&& dis, real* v, size_t n) noexcept
    {
        detail::random_number_generator_impl& rng =
            zi::singleton<detail::random_number_generator_impl>::instance();

        guard g(rng);

        for ( size_t i = 0; i < n; ++i )
        {
            v[i] = dis(rng.rng);
        }
    }

private:
    std::uniform_real_distribution<real> dis;

public:
    uniform_init( real low, real up )
        : dis(low, up)
    {}

    explicit uniform_init( real r = 1 )
        : dis(-r, r)
    {}

    void initialize( real* v, size_t n ) noexcept
    {
        initialize_with_distribution(dis, v, n);
    }

}; // class initializator

}} // namespace znn::fwd
