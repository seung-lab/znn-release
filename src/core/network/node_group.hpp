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

#ifndef ZNN_CORE_NETWORK_NODE_GROUP_HPP_INCLUDED
#define ZNN_CORE_NETWORK_NODE_GROUP_HPP_INCLUDED

#include "../types.hpp"

namespace zi { namespace znn {

class node_group
{
private:
    vec3i size_  = vec3i::zero;
    vec3i stride_= vec3i::zero;

public:
    bool initialize( const vec3i& size, const vec3i& stride )
    {

    }

public:
    virtual ~node_group();

    virtual void receive_featuremap( size_t,
                                     const vol_p<double>& )
    { UNIMPLEMENTED(); }

    virtual void receive_featuremap( size_t,
                                     const cvol_p<double>&,
                                     const cvol_p<double>& )
    { UNIMPLEMENTED(); }

    virtual void receive_featuremap( size_t,
                                     size_t,
                                     const vol_p<complex>& )
    { UNIMPLEMENTED(); }

    virtual void receive_featuremap( size_t,
                                     size_t,
                                     const cvol_p<complex>&,
                                     const cvol_p<complex>& )
    { UNIMPLEMENTED(); }

    virtual void receive_gradient( size_t,
                                   const vol_p<double>& )
    { UNIMPLEMENTED(); }

    virtual void receive_gradient( size_t,
                                   const cvol_p<double>&,
                                   const cvol_p<double>& )
    { UNIMPLEMENTED(); }

    virtual void receive_gradient( size_t,
                                   const vol_p<complex>& )
    { UNIMPLEMENTED(); }

    virtual void receive_gradient( size_t,
                                   const cvol_p<complex>&,
                                   const cvol_p<complex>& )
    { UNIMPLEMENTED(); }


};

}} // namespace zi::znn

#endif // ZNN_CORE_NETWORK_NODE_GROUP_HPP_INCLUDED
