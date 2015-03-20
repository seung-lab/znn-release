#pragma once

#include "../options/options.hpp"

#include <type_traits>
#include <cmath>
#include <string>

namespace znn { namespace v4 { namespace functions {

struct hyperbolic_tangent
{
private:
    double a_ = 1;
    double b_ = 1;
    double b_over_a;

public:
    hyperbolic_tangent( double a = 1, double b = 1 )
        : a_(a), b_(b), b_over_a(b_/a_)
    {}

    double operator()(double x) const noexcept
    {
        return a_ * std::tanh( b_ * x );
    }

    double grad(double f) const noexcept
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
