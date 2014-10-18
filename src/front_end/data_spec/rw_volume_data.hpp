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

#ifndef ZNN_RW_VOLUME_DATA_HPP_INCLUDED
#define ZNN_RW_VOLUME_DATA_HPP_INCLUDED

#include "volume_data.hpp"

namespace zi {
namespace znn {

template<class elemT,class volT>
class rw_volume_data : public volume_data<elemT,volT>
{
public:
	void set_value( vec3i pos, elemT val )
	{
		STRONG_ASSERT((this->gbb_).contains(pos));
		pos -= this->off_; // local coordinate		
		(*(this->vol_))[pos[0]][pos[1]][pos[2]] = val;
	}

	// pos is assumed to be a global coordinate.
	void set_patch( vec3i pos, volT vol )
	{
		STRONG_ASSERT((this->rng_).contains(pos));

		// [08/15/2014]
		// This condition can be relaxed.
		STRONG_ASSERT(size_of(vol) == this->FoV_);

		pos 	-= this->off_;			// local coordinate
		vec3i uc = pos - this->rad_;	// upper corner

		std::size_t ox = uc[0];
	    std::size_t oy = uc[1];
	    std::size_t oz = uc[2];
	    std::size_t sx = this->FoV_[0];
	    std::size_t sy = this->FoV_[1];
	    std::size_t sz = this->FoV_[2];

		(*(this->vol_))[boost::indices[range(ox,ox+sx)][range(oy,oy+sy)][range(oz,oz+sz)]] =
                    (*vol)[boost::indices[range(0,sx)][range(0,sy)][range(0,sz)]];
	}


public:
	rw_volume_data( volT vol,
					const vec3i& FoV = vec3i::zero,
					const vec3i& off = vec3i::zero )
		: volume_data<elemT,volT>(vol,FoV,off)
	{}

	~rw_volume_data()
	{}

}; // class rw_volume_data

typedef rw_volume_data<double,double3d_ptr>	rw_dvolume_data;
typedef rw_volume_data<long_t,long3d_ptr>	rw_lvolume_data;
typedef rw_volume_data<bool,bool3d_ptr>		rw_bvolume_data;

typedef boost::shared_ptr<rw_dvolume_data> 	rw_dvolume_data_ptr;
typedef boost::shared_ptr<rw_lvolume_data> 	rw_lvolume_data_ptr;
typedef boost::shared_ptr<rw_bvolume_data> 	rw_bvolume_data_ptr;

}} // namespace zi::znn

#endif // ZNN_RW_VOLUME_DATA_HPP_INCLUDED