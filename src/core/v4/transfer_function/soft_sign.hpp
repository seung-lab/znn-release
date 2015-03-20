#pragma once

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>

namespace znn { namespace v4 { namespace functions {

struct soft_sign
{
public:
    double operator()(double x) const noexcept
    {
        return x / ( static_cast<double>(1) + std::abs(x) );
    }

    double grad(double f) const noexcept
    {
        return (static_cast<double>(1) - std::abs(f)) *
            (static_cast<double>(1) - std::abs(f));
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "soft_sign");
        return ret;
    }
};

}}} // namespace znn::v4::functions
