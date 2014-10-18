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

#ifndef ZNN_FORWARD_SCANNER_HPP_INCLUDED
#define ZNN_FORWARD_SCANNER_HPP_INCLUDED

#include <string>

namespace zi {
namespace znn {

class forward_scanner
{
protected:
	virtual void load( const std::string& fname ) = 0;

public:
	virtual bool pull( std::list<double3d_ptr>& inputs ) = 0;
	virtual void push( std::list<double3d_ptr>& outputs ) = 0;	
	virtual void save( const std::string& fpath ) const = 0;

}; // abstract class forward_scanner

typedef boost::shared_ptr<forward_scanner> forward_scanner_ptr;

}} // namespace zi::znn

#endif // ZNN_FORWARD_SCANNER_HPP_INCLUDED