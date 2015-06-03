#pragma once

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>

namespace znn { namespace v4 { namespace functions {

struct soft_sign
{
public:
    real operator()(real x) const noexcept
    {
        return x / ( static_cast<real>(1) + std::abs(x) );
    }

    real grad(real f) const noexcept
    {
        return (static_cast<real>(1) - std::abs(f)) *
            (static_cast<real>(1) - std::abs(f));
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "soft_sign");
        return ret;
    }
};

}}} // namespace znn::v4::functions
