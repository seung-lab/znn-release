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