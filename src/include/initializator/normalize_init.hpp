#pragma once

#include "initializator.hpp"
#include <algorithm>

namespace znn { namespace v4 {

class normalize_init: public initializator
{
private:
    real lower_;
    real upper_;

    void do_initialize( real* v, size_t n ) noexcept override
    {
        real min_val = *std::min_element(v,v+n);
        real max_val = *std::max_element(v,v+n);

        real old_range = max_val - min_val;
        real new_range = upper_  - lower_;

        if ( old_range < std::numeric_limits<real>::epsilon() )
        {
            old_range = std::numeric_limits<real>::max();
        }

        for ( std::size_t i = 0; i < n; ++i )
        {
            v[i] = new_range * (v[i] - min_val) / old_range + lower_;
        }

    }

public:
    explicit normalize_init( real low = 0, real up = 1 )
        : lower_(low)
        , upper_(up)
    {}

}; // class normalize_init

}} // namespace znn::v4
