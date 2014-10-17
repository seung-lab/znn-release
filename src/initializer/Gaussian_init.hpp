#ifndef ZNN_GAUSSIAN_INIT_HPP_INCLUDED
#define ZNN_GAUSSIAN_INIT_HPP_INCLUDED

#include "initializer.hpp"

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
