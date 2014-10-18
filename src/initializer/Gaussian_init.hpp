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

#ifndef ZNN_GAUSSIAN_INIT_HPP_INCLUDED
#define ZNN_GAUSSIAN_INIT_HPP_INCLUDED

#include "initializer.hpp"
#include "../core/volume_utils.hpp"

#include <boost/random/normal_distribution.hpp>

namespace zi {
namespace znn {

class Gaussian_init : virtual public initializer
{
private:
	typedef boost::normal_distribution<> norm_dist;
	typedef boost::variate_generator<boost::mt19937&, norm_dist> generator;

private:
	double 	mu;		// mean
	double 	sigma;	// standard deviation

public:    
    virtual void initialize( double3d_ptr w )
    {
    	volume_utils::zero_out(w);
    	norm_dist dist(mu,sigma);
	    generator gen(rng,dist);
		volume_utils::random_initialization(gen,w);
    }

    virtual void init( const std::string& params )
    {
    	// parser for parsing arguments
		std::vector<double> args;
		zi::zargs_::parser<std::vector<double> > arg_parser;
		
		bool parsed = arg_parser.parse(&args,params);
		if ( parsed && (args.size() >= 1) )
		{
			mu = args[0];
			if ( args.size() >= 2 )
			{
				sigma = args[1];
			}
		}
    }

public:
	Gaussian_init( double _mu = 0.0, 
				   double _sigma = 0.01 )
		: mu(_mu)
		, sigma(_sigma)
	{}

}; // class Gaussian_init

}} // namespace zi::znn

#endif // ZNN_GAUSSIAN_INIT_HPP_INCLUDED