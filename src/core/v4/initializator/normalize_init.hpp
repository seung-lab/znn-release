#pragma once

#include "initializator.hpp"
#include <algorithm>

namespace znn { namespace v4 {

class normalize_init: public initializator
{
private:
    double lower_;
    double upper_;

    void do_initialize( double* v, size_t n ) noexcept override
    {
        double min_val = *std::min_element(v,v+n);
        double max_val = *std::max_element(v,v+n);

        double old_range = max_val - min_val;
        double new_range = upper_  - lower_;

        if ( old_range < std::numeric_limits<double>::epsilon() )
        {
            old_range = std::numeric_limits<double>::max();
        }

        for ( std::size_t i = 0; i < n; ++i )
        {
            v[i] = new_range * (v[i] - min_val) / old_range + lower_;
        }

    }

public:
    explicit normalize_init( double low = 0, double up = 1 )
        : lower_(low)
        , upper_(up)
    {}

}; // class normalize_init

}} // namespace znn::v4
