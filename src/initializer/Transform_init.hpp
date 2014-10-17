#ifndef ZNN_TRANSFORM_INIT_HPP_INCLUDED
#define ZNN_TRANSFORM_INIT_HPP_INCLUDED

#include "initializer.hpp"


namespace zi {
namespace znn {

class Transform_init : virtual public initializer
{
private:
	double 	lower;
	double 	upper;

public:    
    virtual void initialize( double3d_ptr w )
    {
    	volume_utils::transform(w,upper,lower);
    }

    virtual void init( const std::string& params )
    {
    	// parser for parsing arguments
		std::vector<double> args;
		zi::zargs_::parser<std::vector<double> > arg_parser;

		bool parsed = arg_parser.parse(&args,params);
		if ( parsed && (args.size() >= 2) )
		{
			lower = args[0];
			upper = args[1];			
		}
    }


public:
	Transform_init( double _lower = static_cast<double>(0),
				 	double _upper = static_cast<double>(1) )
		: lower(_lower)
		, upper(_upper)
	{}

}; // class Transform_init

}} // namespace zi::znn

#endif // ZNN_TRANSFORM_INIT_HPP_INCLUDED
