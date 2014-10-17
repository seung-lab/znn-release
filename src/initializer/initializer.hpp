#ifndef ZNN_INITIALIZER_HPP_INCLUDED
#define ZNN_INITIALIZER_HPP_INCLUDED

#include "../core/types.hpp"

#include <boost/random.hpp>

namespace zi {
namespace znn {

class initializer
{
public:    
    virtual void initialize( double3d_ptr w ) = 0;
    virtual void init( const std::string& params ) = 0;

}; // abstract class initializer

typedef boost::shared_ptr<initializer> initializer_ptr;

// global random number generator
namespace {
boost::mt19937 rng = boost::mt19937(time(0));
} // anonymous namespace

}} // namespace zi::znn

#endif // ZNN_INITIALIZER_HPP_INCLUDED
