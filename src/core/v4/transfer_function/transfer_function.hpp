#pragma once

#include "../types.hpp"
#include "../assert.hpp"
#include "../cube/cube.hpp"
#include "../options/options.hpp"

#include <memory>
#include <map>
#include <string>

namespace znn { namespace v4 {

class transfer_function_interface
{
public:
    virtual ~transfer_function_interface() {}
    virtual void apply(cube<double>&) noexcept = 0;
    virtual void apply(cube<double>&, double) noexcept = 0;
    virtual void apply_grad(cube<double>&, const cube<double>&) noexcept = 0;
    virtual options serialize() const = 0;
};


template<class F>
using grad_result_t =
    decltype(std::declval<const F&>().grad(std::declval<double>()));

template<class F, class = void>
struct has_public_member_grad: std::false_type {};

template<class F>
struct has_public_member_grad< F, void_t<grad_result_t<F>> >
    : std::is_same< grad_result_t<F>, double > {};


template<class F>
class transfer_function_wrapper: public transfer_function_interface
{
private:
    F f_;

    template<typename T>
    typename std::enable_if<has_public_member_grad<T>::value>::type
    apply_grad(cube<double>& g, const cube<double>& f, const T& fn) noexcept
    {
        double* gp = g.data();
        const double* fp = f.data();
        size_t  n = g.num_elements();
        for ( size_t i = 0; i < n; ++i )
            gp[i] *= fn.grad(fp[i]);
    }

    template<typename T>
    typename std::enable_if<!has_public_member_grad<T>::value>::type
    apply_grad(cube<double>&, const cube<double>&, const T&) noexcept
    {}

public:
    explicit transfer_function_wrapper(F f = F())
        : f_(f)
    {}

    void apply(cube<double>& v) noexcept override
    {
        double* d = v.data();
        size_t  n = v.num_elements();
        for ( size_t i = 0; i < n; ++i )
            d[i] = f_(d[i]);
    }

    void apply(cube<double>& v, double bias) noexcept override
    {
        double* d = v.data();
        size_t  n = v.num_elements();
        for ( size_t i = 0; i < n; ++i )
            d[i] = f_(d[i] + bias);
    }

    void apply_grad(cube<double>& g, const cube<double>& f) noexcept override
    {
        ZI_ASSERT(size(g)==size(f));
        apply_grad(g,f,f_);
    }

    options serialize() const
    {
        return f_.serialize();
    }

};

class transfer_function
{
private:
    std::shared_ptr<transfer_function_interface> f_;

public:
    transfer_function()
        : f_()
    {}

    explicit operator bool() const
    {
        return static_cast<bool>(f_);
    }

    template<typename F>
    transfer_function(const F& f)
        : f_(new transfer_function_wrapper<F>(f))
    {}

    template<typename F>
    transfer_function& operator=(const F& f)
    {
        f_ = std::shared_ptr<transfer_function_interface>
            (new transfer_function_wrapper<F>(f));
        return *this;
    }

    void apply(cube<double>& v) noexcept
    {
        if ( f_ ) f_->apply(v);
    }

    void apply(cube<double>& v, double bias) noexcept
    {
        if ( f_ ) f_->apply(v, bias);
    }

    void apply_grad(cube<double>& g, const cube<double>& f) noexcept
    {
        if ( f_ ) f_->apply_grad(g,f);
    }

    options serialize() const
    {
        if ( f_ )
        {
            return f_->serialize();
        }
        else
        {
            return options();
        }
    }

};

transfer_function get_transfer_function( const options& op )
{
    std::string fn = op.require_as<std::string>("function");

    if ( fn == "tanh" )
    {
        ovec2d p = op.optional_as<ovec2d>("function_args", "1,1");
        return transfer_function(functions::hyperbolic_tangent(p[0],p[1]));
    }
    else if ( fn == "linear" )
    {
        ovector<double> p =
            op.optional_as<ovector<double>>("function_args", "1,0");

        ZI_ASSERT(p.size()&&p.size()<3);

        if ( p.size() == 1 )
        {
            return transfer_function(functions::linear(p[0]));
        }
        else if ( p.size() == 2 )
        {
            return transfer_function(functions::linear(p[0], p[1]));
        }
    }
    else if ( fn == "rectify_linear" )
    {
        return transfer_function(functions::rectify_linear());
    }
    else if ( fn == "soft_sign" )
    {
        return transfer_function(functions::soft_sign());
    }
    else if ( fn == "logistics" )
    {
        return transfer_function(functions::logistics());
    }
    else if ( fn == "forward_logistics" )
    {
        return transfer_function(functions::forward_logistics());
    }

    throw std::logic_error(HERE() + "unknown function: " + fn);
}

}} // namespace znn::v4
