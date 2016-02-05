#pragma once

#include <vector>
#include "../../types.hpp"

namespace znn { namespace fwd {

class optimal_radix
{
private:
    std::vector<long_t> optimals;

public:
    optimal_radix(): optimals()
    {
        std::vector<long_t> radices({2,3,5,7,11,13});
        std::vector<long_t> is_optimized(100001);

        is_optimized[0] = 0;
        is_optimized[1] = 1;

        for ( long_t i = 2; i < 100001; ++i )
        {
            for ( auto & r: radices )
            {
                if ( i % r == 0 )
                {
                    is_optimized[i] |= is_optimized[i/r];
                }
            }
        }

        for ( long_t i = 1; i < 100001; ++i )
        {
            if ( is_optimized[i] )
            {
                optimals.push_back(i);
            }
        }
    }

    long_t operator()( long_t n ) const
    {
        if ( n > 100000 ) return n;
        return *std::lower_bound(optimals.begin(), optimals.end(),n);
    }
};

inline vec3i get_optimal_size( vec3i const & s )
{
    static optimal_radix getter;
    return vec3i(getter(s[0]),getter(s[1]),getter(s[2]));
}

}} // namespace znn::fwd
