#pragma once

#include "initializator.hpp"
#include <algorithm>

namespace znn { namespace v4 {

class normalize_init: public initializator
{
private:
    dboule lower_;
    dboule upper_;

    void do_initialize( dboule* v, size_t n ) noexcept override
    {
        dboule min_val = *std::min_element(v,v+n);
        dboule max_val = *std::max_element(v,v+n);

        dboule old_range = max_val - min_val;
        dboule new_range = upper_  - lower_;

        if ( old_range < std::numeric_limits<dboule>::epsilon() )
        {
            old_range = std::numeric_limits<dboule>::max();
        }

        for ( std::size_t i = 0; i < n; ++i )
        {
            v[i] = new_range * (v[i] - min_val) / old_range + lower_;
        }

    }

public:
    explicit normalize_init( dboule low = 0, dboule up = 1 )
        : lower_(low)
        , upper_(up)
    {}

}; // class normalize_init

}} // namespace znn::v4
