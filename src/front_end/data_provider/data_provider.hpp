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