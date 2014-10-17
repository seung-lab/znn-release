#ifndef ZNN_DATA_PROVIDER_HPP_INCLUDED
#define ZNN_DATA_PROVIDER_HPP_INCLUDED

#include "input_sampler.hpp"

#include <string>

namespace zi {
namespace znn {

class data_provider
{
protected:
	virtual void load( const std::string& fname ) = 0;

public:
	// sequential or random sampling
	virtual input_sampler_ptr next_sample() = 0;
	virtual input_sampler_ptr first_sample() = 0;

	// random sampling
	virtual input_sampler_ptr random_sample() = 0;

}; // abstract class data_provider

typedef boost::shared_ptr<data_provider> data_provider_ptr;

}} // namespace zi::znn

#endif // ZNN_DATA_PROVIDER_HPP_INCLUDED