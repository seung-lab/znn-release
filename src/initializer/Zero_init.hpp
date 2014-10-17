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
