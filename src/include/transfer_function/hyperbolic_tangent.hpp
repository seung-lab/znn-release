#pragma once

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>

namespace znn { namespace v4 { namespace functions {

struct hyperbolic_tangent
{
private:
    real a_ = 1;
    real b_ = 1;
    real b_over_a;

public:
    hyperbolic_tangent( real a = 1, real b = 1 )
        : a_(a), b_(b), b_over_a(b_/a_)
    {}

    real operator()(real x) const noexcept
    {
        return a_ * std::tanh( b_ * x );
    }

    real grad(real f) const noexcept
    {
        return b_over_a * ( a_ - f ) * ( a_ + f );
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "tanh").
            push("function_args", std::to_string(a_) + "," + std::to_string(b_));
        return ret;
    }

};

}}} // namespace znn::v4functions
