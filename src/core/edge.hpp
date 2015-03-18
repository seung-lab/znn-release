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

#ifndef ZNN_CORE_EDGE_HPP_INCLUDED
#define ZNN_CORE_EDGE_HPP_INCLUDED

#include "types.hpp"
#include "utils.hpp"
#include "filter.hpp"

namespace zi {
namespace znn {


class nodes;

template<typename E>
class znn_edgee
{
private:
    static_assert(!E::works_on_ffts_tag::value, "Use znn_fft_edge<T>");

private:
    nodes* in_     ;
    size_t in_n_   ;
    nodes* out_    ;
    size_t out_n_  ;
    bool   no_back_;

    template<typename Dummy = E>
    if_t<E::async_forward::value&&E::is_convolving::value>
    forward_callback( const cvol_p<double>& f, const cvol_p<double>& w,
                      const vec3i& sparse = vec3i::one )
    {
        out_->receive_featuremap(out_n_, f, w, sparse);
    }

    template<typename Dummy = E>
    if_t<E::async_forward::value&&!E::is_convolving::value>
    forward_callback( vol_p<double>&& f )
    {
        out_->receive_featuremap(out_n_, std::move(f));
    }


    template<typename Dummy = E>
    if_t<E::async_backward::value&&E::is_convolving::value>
    forward_callback( const cvol_p<double>& f, const cvol_p<double>& w,
                      const vec3i& sparse = vec3i::one )
    {
        if ( no_back_ )
        {
            in_->receive_gradient(in_n_, f, w, sparse);
        }

    }

    template<typename Dummy = E>
    if_t<E::async_backward::value&&!E::is_convolving::value>
    forward_callback( vol_p<double>&& f )
    {
        in_->receive_gradient(in_n_, std::move(f));
    }


};



}} // namespace zi::znn


#endif //  ZNN_CORE_EDGE_HPP_INCLUDED
