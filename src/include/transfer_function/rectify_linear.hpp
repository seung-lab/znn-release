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
    real operator()(real x) const noexcept
    {
        return std::max( static_cast<real>(0), x );
    }

    real grad(real f) const noexcept
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
