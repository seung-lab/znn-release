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

#ifndef ZNN_AFFINITY_DATA_PROVIDER_HPP_INCLUDED
#define ZNN_AFFINITY_DATA_PROVIDER_HPP_INCLUDED

#include "volume_data_provider.hpp"
#include "transformer/affinity_transformer.hpp"

namespace zi {
namespace znn {

class affinity_data_provider : virtual public volume_data_provider
{
// data augmentation
public:
	void data_augmentation( bool data_aug = false )
	{
		if ( data_aug )
		{
			trans_ = transformer_ptr(new affinity_transformer());
		}
		else
		{
			trans_.reset();
		}
	}


protected:
	virtual void add_label( dvolume_data_ptr lbl )
	{
		std::size_t idx = lbls_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		vec3i FoV = out_szs_[idx];
		vec3i sft = vec3i::zero;

		// for affinity graph transformation
		FoV += vec3i::one;
		sft += vec3i::one;
		
		lbl->set_FoV(FoV,sft);
		lbls_.push_back(lbl);
	}

	virtual void add_mask( bvolume_data_ptr msk )
	{
		std::size_t idx = msks_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		vec3i FoV = out_szs_[idx];
		vec3i sft = vec3i::zero;

		// for affinity graph transformation
		FoV += vec3i::one;
		sft += vec3i::one;
		
		msk->set_FoV(FoV,sft);
		msks_.push_back(msk);
	}


// constructor & destructor
public:
	affinity_data_provider( const std::string& fname, 
						    std::vector<vec3i> in_szs,
						    std::vector<vec3i> out_szs )
		: volume_data_provider()
	{
		in_szs_ = in_szs;
		out_szs_ = out_szs;

		load(fname);
		init();
	}

	virtual ~affinity_data_provider()
	{}

}; // class affinity_data_provider

typedef boost::shared_ptr<affinity_data_provider> affinity_data_provider_ptr;

}} // namespace zi::znn

#endif // ZNN_AFFINITY_DATA_PROVIDER_HPP_INCLUDED