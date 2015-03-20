#pragma once

#include "../options/options.hpp"

#include <string>
#include <type_traits>

namespace znn { namespace v4 { namespace functions {

struct linear
{
private:
    double a_ = 1;
    double b_ = 0;

public:
    linear( double a = 1, double b = 0 )
        : a_(a), b_(b)
    {}

    double operator()(double x) const noexcept
    {
        return a_ * x + b_;
    }

    double grad(double) const noexcept
    {
        return a_;
    }

    options serialize() const
    {
        options ret;
        ret.push("function", "linear").
            push("function_args", std::to_string(a_) + "," + std::to_string(b_));
        return ret;
    }

};

}}} // namespace znn::v4::functions
