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

#ifndef ZNN_SAMPLE_HPP_INCLUDED
#define ZNN_SAMPLE_HPP_INCLUDED

namespace zi {
namespace znn {

class sample
{
public:
	std::list<double3d_ptr>		inputs;
	std::list<double3d_ptr> 	labels;
	std::list<double3d_ptr>		wmasks;
	std::list<bool3d_ptr>		masks ;

public:
	// for debugging purpose
	void save( const std::string& fname )
	{
		volume_utils::save_list(inputs, fname + ".input");
		volume_utils::save_list(labels, fname + ".label");
		volume_utils::save_list(wmasks, fname + ".wmask");
		volume_utils::save_list(masks,  fname + ".mask" );
	}

public:
	sample( std::list<double3d_ptr> i,
		   	std::list<double3d_ptr> l,
		   	std::list<double3d_ptr> w,
		   	std::list<bool3d_ptr>   m )
		: inputs(i)
		, labels(l)
		, wmasks(w)
		, masks(m)
	{}

}; // class sample

typedef boost::shared_ptr<sample> sample_ptr;

}} // namespace zi::znn

#endif // ZNN_SAMPLE_HPP_INCLUDED