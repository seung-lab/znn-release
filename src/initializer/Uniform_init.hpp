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

#ifndef ZNN_UNIFORM_INIT_HPP_INCLUDED
#define ZNN_UNIFORM_INIT_HPP_INCLUDED

#include "initializer.hpp"
#include "../core/volume_utils.hpp"

#include <boost/random/uniform_real.hpp>

namespace zi {
namespace znn {

class Uniform_init : virtual public initializer
{
private:
	typedef boost::uniform_real<> uniform_dist;
	typedef boost::variate_generator<boost::mt19937&, uniform_dist> generator;

private:
	double	lower;
	double	upper;

public:    
    virtual void initialize( double3d_ptr w )
    {
    	volume_utils::zero_out(w);
    	uniform_dist dist(lower,upper);
	    generator gen(rng,dist);	    
		volume_utils::random_initialization(gen,w);
    }

    virtual void init( const std::string& params )
    {
    	// parser for parsing arguments
		std::vector<double> args;
		zi::zargs_::parser<std::vector<double> > arg_parser;
		
		bool parsed = arg_parser.parse(&args,params);
		if ( parsed )
		{
			if ( args.size() == 1 )
			{
				lower = -std::abs(args[0]);
				upper = std::abs(args[0]);
			}
			else if ( args.size() == 2 )
			{
				lower = std::min(args[0],args[1]);
				upper = std::max(args[0],args[1]);
			}
		}
    }

public:
	Uniform_init( double _lower = -0.1, 
				  double _upper =  0.1 )
		: lower(_lower)
		, upper(_upper)
	{}

}; // class Uniform_init

}} // namespace zi::znn

#endif // ZNN_UNIFORM_INIT_HPP_INCLUDED
