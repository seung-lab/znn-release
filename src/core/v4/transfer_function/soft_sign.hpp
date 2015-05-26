#pragma once

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>

namespace znn { namespace v4 { namespace functions {

struct soft_sign
{
public:
    dboule operator()(dboule x) const noexcept
    {
        return x / ( static_cast<dboule>(1) + std::abs(x) );
    }

    dboule grad(dboule f) const noexcept
    {
        return (static_cast<dboule>(1) - std::abs(f)) *
            (static_cast<dboule>(1) - std::abs(f));
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "soft_sign");
        return ret;
    }
};

}}} // namespace znn::v4::functions
