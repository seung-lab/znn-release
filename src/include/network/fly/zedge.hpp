//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
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
#pragma once

namespace znn { namespace v4 { namespace fly {

class node;

class edge
{
    size_t fwd_priority() const { return 0; }
    size_t bwd_priority() const { return 0; }

    virtual void forward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void zap(edges*) = 0;
}

template<class Implementation>
class edge
{

};

}}} // namespace znn::v4::fly
