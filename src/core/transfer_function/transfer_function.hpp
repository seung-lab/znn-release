//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZNN_CORE_TRANSFER_FUNCTION_TRANSFER_FUNCTION_HPP_INCLUDED
#define ZNN_CORE_TRANSFER_FUNCTION_TRANSFER_FUNCTION_HPP_INCLUDED

#include "../types.hpp"
#include "../utils.hpp"

#include <memory>

namespace zi {
namespace znn {

class transfer_function_interface
{
public:
    virtual void apply(vol<double>&) noexcept = 0;
    virtual void apply(vol<double>&, double) noexcept = 0;
    virtual void apply_grad(vol<double>&, const vol<double>&) noexcept = 0;
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
    apply_grad(vol<double>& g, const vol<double>& f, const T& fn) noexcept
    {
        double* gp = g.data();
        const double* fp = f.data();
        size_t  n = g.num_elements();
        for ( size_t i = 0; i < n; ++i )
            gp[i] *= fn.grad(fp[i]);
    }

    template<typename T>
    typename std::enable_if<!has_public_member_grad<T>::value>::type
    apply_grad(vol<double>&, const vol<double>&, const T&) noexcept
    {}

public:
    explicit transfer_function_wrapper(F f = F())
        : f_(f)
    {}

    void apply(vol<double>& v) noexcept override
    {
        double* d = v.data();
        size_t  n = v.num_elements();
        for ( size_t i = 0; i < n; ++i )
            d[i] = f_(d[i]);
    }

    void apply(vol<double>& v, double bias) noexcept override
    {
        double* d = v.data();
        size_t  n = v.num_elements();
        for ( size_t i = 0; i < n; ++i )
            d[i] = f_(d[i] + bias);
    }

    void apply_grad(vol<double>& g, const vol<double>& f) noexcept override
    {
        ZI_ASSERT(size(g)==size(f));
        apply_grad(g,f,f_);
    }
};

class transfer_function
{
private:
    std::unique_ptr<transfer_function_interface> f_;

public:
    transfer_function()
        : f_()
    {}

    template<typename F>
    transfer_function(const F& f)
        : f_(new transfer_function_wrapper<F>(f))
    {}

    template<typename F>
    transfer_function& operator=(const F& f)
    {
        f_ = std::unique_ptr<transfer_function_interface>
            (new transfer_function_wrapper<F>(f));
        return *this;
    }

    void apply(vol<double>& v) noexcept
    {
        f_->apply(v);
    }

    void apply(vol<double>& v, double bias) noexcept
    {
        f_->apply(v, bias);
    }

    void apply_grad(vol<double>& g, const vol<double>& f) noexcept
    {
        f_->apply_grad(g,f);
    }
};


}} // namespace zi::znn


#endif // ZNN_CORE_TRANSFER_FUNCTION_TRANSFER_FUNCTION_HPP_INCLUDED
