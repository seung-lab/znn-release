#ifndef ZNN_CONSTANT_INIT_HPP_INCLUDED
#define ZNN_CONSTANT_INIT_HPP_INCLUDED

#include "initializer.hpp"


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
