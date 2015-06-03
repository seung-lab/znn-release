#pragma once

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>

namespace znn { namespace v4 { namespace functions {

struct logistics
{
    real operator()(real x) const noexcept
    {
        return static_cast<real>(1) / (static_cast<real>(1) + std::exp(-x));
    }

    real grad(real f) const noexcept
    {
        return f * (static_cast<real>(1) - f);
    }


    options serialize() const
    {
        options ret;
        ret.push("function", "logistics");
        return ret;
    }


};


struct forward_logistics
{
    real operator()(real x) const noexcept
    {
        return static_cast<real>(1) / (static_cast<real>(1) + std::exp(-x));
    }


    options serialize() const
    {
        options ret;
        ret.push("function", "forward_logistics");
        return ret;
    }

};

}}} // namespace znn::v4::functions
