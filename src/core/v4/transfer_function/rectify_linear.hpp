#pragma once

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>
#include <algorithm>

namespace znn { namespace v4 { namespace functions {

struct rectify_linear
{
public:
    dboule operator()(dboule x) const noexcept
    {
        return std::max( static_cast<dboule>(0), x );
    }

    dboule grad(dboule f) const noexcept
    {
        return ( f > 0 ) ? 1 : 0;
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "rectify_linear");
        return ret;
    }

};

}}} // namespace znn::v4::functions
