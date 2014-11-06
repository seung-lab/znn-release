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

#ifndef ZNN_VOLUME_DATA_HPP_INCLUDED
#define ZNN_VOLUME_DATA_HPP_INCLUDED

#include "../../core/utils.hpp"
#include "../../core/volume_utils.hpp"
#include "box.hpp"


namespace zi {
namespace znn {

template<class elemT,class volT>
class volume_data
{
protected:
	volT	vol_;	// volume
	vec3i	dim_;	// dimension of volume

	vec3i	FoV_;	// field of View
	vec3i	rad_;	// radius of FoV
	vec3i	off_;	// offset w.r.t. global origin	
	
	// When FoV is not symmetric (even-sized, e.g., [6 6 6]), 
	// it determines how the center of FoV is shifted
	vec3i	sft_;
	vec3i 	top_;	// top margin
	vec3i	btm_;	// bottom margin

	box		gbb_;	// global bounding box
	box		rng_;	// range


public:
	elemT get_value( vec3i pos ) const
	{
		// validity check
		STRONG_ASSERT(gbb_.contains(pos));

		pos -= off_; // local coordinate

		return (*vol_)[pos[0]][pos[1]][pos[2]];
	}
	
	volT get_patch( vec3i pos ) const
	{
		// validity check
		STRONG_ASSERT(rng_.contains(pos));

		pos -= off_; // local coordinate
		vec3i uc = pos - top_;
		return volume_utils::crop(vol_,uc,FoV_);
	}

	const volT& get_volume() const
	{
		return vol_;
	}

	// global range
	const box& get_range() const
	{
		return rng_;
	}

	const vec3i& get_dimemsion() const
	{
		return dim_;
	}

	const vec3i& get_FoV() const
	{
		return FoV_;
	}

	const vec3i& get_offset() const
	{
		return off_;
	}

	const vec3i& get_shift() const
	{
		return sft_;
	}


public:
	// void mirror( std::size_t w )
	// {
	// 	STRONG_ASSERT( dim_ != vec3i::zero );
	// 	set_data(volume_utils::mirror_boundary(vol_,w));
	// }


private:
	void set_range()
	{
		vec3i uc = off_ + top_;			// upper corner
		vec3i lc = off_ + dim_ - btm_;	// lower corner

		rng_ = box(uc,lc);
	}

	void set_global_bounding_box()
	{
		gbb_ = box(vec3i::zero,dim_) + off_;
	}


public:
	void set_data( volT vol )
	{
		vol_ = vol;
		dim_ = size_of(vol_);
		
		set_global_bounding_box();
		set_range();
	}

	void set_FoV( const vec3i& FoV, const vec3i& sft = vec3i::zero )
	{
		FoV_ = (FoV == vec3i::zero ? dim_ : FoV);

		if ( FoV_[0] > dim_[0] ) FoV_[0] = dim_[0];
		if ( FoV_[1] > dim_[1] ) FoV_[1] = dim_[1];
		if ( FoV_[2] > dim_[2] ) FoV_[2] = dim_[2];
		
		rad_ = FoV_/vec3i(2,2,2);

		// set_range() included
		set_shift(sft);		
	}

	void set_offset( const vec3i& off )
	{
		off_ = off;

		set_global_bounding_box();
		set_range();
	}

	void set_shift( const vec3i& sft )
	{
		sft_ = sft;
		if ( sft_[0] > 0 ) sft_[0] = 1;
		if ( sft_[1] > 0 ) sft_[1] = 1;
		if ( sft_[2] > 0 ) sft_[2] = 1;

		// top margin
		top_ = rad_;
		if ( FoV_[0] % 2 == 0 ) top_[0] -= sft_[0];
        if ( FoV_[1] % 2 == 0 ) top_[1] -= sft_[1];
        if ( FoV_[2] % 2 == 0 ) top_[2] -= sft_[2];

        // bottom margin
        btm_ = FoV_ - top_ - vec3i::one;

		set_range();
	}


public:
	void print() const
	{
		std::cout << "[volume_data]" << std::endl;
		std::cout << "dim: " << dim_ << std::endl;
		std::cout << "FoV: " << FoV_ << std::endl;
		// std::cout << "rad: " << rad_ << std::endl;
		std::cout << "off: " << off_ << std::endl;
		// std::cout << "sft: " << sft_ << std::endl;
		// std::cout << "top: " << top_ << std::endl;
		// std::cout << "btm: " << btm_ << std::endl;
		// std::cout << "gbb: " << gbb_ << std::endl;
		// std::cout << "rng: " << rng_ << std::endl;
	}

	void save( const std::string& fpath ) const
	{
		volume_utils::save(vol_,fpath);
		export_size_info(dim_,fpath);
	}


public:
	volume_data( volT vol,
				 const vec3i& FoV = vec3i::zero,
				 const vec3i& off = vec3i::zero,
				 const vec3i& sft = vec3i::zero )
		: vol_(vol)
		, dim_(size_of(vol))
		, FoV_(FoV)
		, rad_(FoV/vec3i(2,2,2))		
		, off_(off)
		, sft_(sft)
		, gbb_(box(vec3i::zero,size_of(vol))+off)
		, rng_()
	{
		STRONG_ASSERT(dim_ != vec3i::zero);
		set_FoV(FoV,sft);	// including set_range() inside
		set_global_bounding_box();
	}

	~volume_data()
	{}

}; // class volume_data

typedef volume_data<double,double3d_ptr>	dvolume_data;
typedef volume_data<long_t,long3d_ptr>		lvolume_data;
typedef volume_data<bool,bool3d_ptr>		bvolume_data;

typedef boost::shared_ptr<dvolume_data> 	dvolume_data_ptr;
typedef boost::shared_ptr<lvolume_data> 	lvolume_data_ptr;
typedef boost::shared_ptr<bvolume_data> 	bvolume_data_ptr;

}} // namespace zi::znn

#endif // ZNN_VOLUME_DATA_HPP_INCLUDED