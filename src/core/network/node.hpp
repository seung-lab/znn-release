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

#ifndef ZNN_CORE_NETWORK_NODE_HPP_INCLUDED
#define ZNN_CORE_NETWORK_NODE_HPP_INCLUDED

#include "../types.hpp"
#include "../volume_operators.hpp"
#include "../transfer_function/transfer_function.hpp"
#include "../bias.hpp"

namespace zi {
namespace znn {

class znn_node
{
public:
    virtual ~znn_node();
    virtual cvol_p<double> forward (const vol_p<double>&) = 0;
    virtual cvol_p<double> backward(const vol_p<double>&) = 0;
};

template<typename F>
class transfer_function_serial_node
{
private:
    bias&                        bias_      ;
    transfer_function_wrapper<F> f_         ;
    double                       patch_sz_  ;

    vol_p<double>                fmap_      ;
public:
    transfer_function_serial_node( bias& b, F f = F(), double ps = 1)
        : bias_(b), f_(f), patch_sz_(ps) {}

    cvol_p<double> forward(const vol_p<double>& f)
    {
        ZI_ASSERT(!fmap_);
        f_.apply(*f, bias_.b());
        fmap_ = f;
        return f;
    }

    cvol_p<double> backward(const vol_p<double>& g)
    {
        ZI_ASSERT(fmap_);
        f_.apply_grad(g, fmap_);
        fmap_.reset();
        double dEdW = sum(*g);
        bias_.update(dEdW, patch_sz_);
        return g;
    }
};

class node_group
{
public:
    virtual ~node_group();

    virtual void receive_featuremap( size_t,
                                     const vol_p<double>& )
    { DIE(); }

    virtual void receive_featuremap( size_t,
                                     const cvol_p<double>&,
                                     const cvol_p<double>& )
    { DIE(); }

    virtual void receive_featuremap( size_t,
                                     const vol_p<complex>& )
    { DIE(); }

    virtual void receive_featuremap( size_t,
                                     const cvol_p<complex>&,
                                     const cvol_p<complex>& )
    { DIE(); }

    virtual void receive_gradient( size_t,
                                   const vol_p<double>& )
    { DIE(); }

    virtual void receive_gradient( size_t,
                                   const cvol_p<double>&,
                                   const cvol_p<double>& )
    { DIE(); }

    virtual void receive_gradient( size_t,
                                   const vol_p<complex>& )
    { DIE(); }

    virtual void receive_gradient( size_t,
                                   const cvol_p<complex>&,
                                   const cvol_p<complex>& )
    { DIE(); }


};

}} // namespace zi::znn

#endif // ZNN_CORE_NETWORK_NODE_HPP_INCLUDED
