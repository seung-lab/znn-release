//
// Copyright (C) 2014  Kisuk Lee <kisuklee@mit.edu>
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

#ifndef ZNN_ZERO_INIT_HPP_INCLUDED
#define ZNN_ZERO_INIT_HPP_INCLUDED

#include "initializer.hpp"

namespace zi {
namespace znn {

class Zero_init : virtual public initializer
{
public:    
    virtual void initialize( double3d_ptr w )
    {
    	volume_utils::zero_out(w);
    }

    virtual void init( const std::string& /* params */ ) {}

}; // class Zero_init

}} // namespace zi::znn

#endif // ZNN_ZERO_INIT_HPP_INCLUDED
