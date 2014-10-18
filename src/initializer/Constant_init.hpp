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

#ifndef ZNN_CONSTANT_INIT_HPP_INCLUDED
#define ZNN_CONSTANT_INIT_HPP_INCLUDED

#include "initializer.hpp"
#include "../core/volume_utils.hpp"

namespace zi {
namespace znn {

class Constant_init : virtual public initializer
{
private:
	double c;

public:    
    virtual void initialize( double3d_ptr w )
    {
    	volume_utils::fill_one(w);
		volume_utils::elementwise_mul_by(w,c);
    }

    virtual void init( const std::string& params )
    {
    	// parser for parsing arguments
		std::vector<double> args;
		zi::zargs_::parser<std::vector<double> > arg_parser;
		
		bool parsed = arg_parser.parse(&args,params);
		if ( parsed && (args.size() >= 1) )
		{
			c = args[0];
		}
    }

public:
	Constant_init( double _c = static_cast<double>(0) )
		: c(_c)
	{}

}; // class Constant_init

}} // namespace zi::znn

#endif // ZNN_CONSTANT_INIT_HPP_INCLUDED