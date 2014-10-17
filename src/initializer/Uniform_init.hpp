#ifndef ZNN_UNIFORM_INIT_HPP_INCLUDED
#define ZNN_UNIFORM_INIT_HPP_INCLUDED

#include "initializer.hpp"

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
