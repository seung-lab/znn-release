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
// load
protected:
	virtual void load( const std::string& fname )
	{
		data_spec_parser parser(fname);

		// inputs
		FOR_EACH( it, parser.input_specs )
		{
			std::cout << "Loading input [" << (*it)->name << "]" << std::endl;
			std::list<dvolume_data_ptr> vols = data_builder::build_volume(*it);
			FOR_EACH( jt, vols )
			{
				add_image(*jt);
			}
			std::cout << std::endl;
		}
			
		// labels
		FOR_EACH( it, parser.label_specs )
		{
			// [kisuklee] Clumsy implementation. Should modify later.
			if ( (*it)->pptype == "affinity" )
			{
				std::cout << "Loading label [" << (*it)->name << "]" << std::endl;
				std::list<dvolume_data_ptr> vols = data_builder::build_volume(*it);
				
				STRONG_ASSERT(vols.size() == 3);
				
				dvolume_data_ptr xaff = vols.front(); vols.pop_front();
				dvolume_data_ptr yaff = vols.front(); vols.pop_front();
				dvolume_data_ptr zaff = vols.front(); vols.pop_front();

				add_label(xaff, vec3i(1,1,0));
				add_label(yaff, vec3i(1,1,0));
				add_label(zaff, vec3i(0,0,1));

				std::cout << std::endl;
			}
		}

		// masks
		FOR_EACH( it, parser.mask_specs )
		{
			// [kisuklee] Clumsy implementation. Should modify later.
			if ( (*it)->pptype == "affinity" || (*it)->pptype == "one" )
			{
				std::cout << "Loading mask [" << (*it)->name << "]" << std::endl;			
				std::list<bvolume_data_ptr> vols = data_builder::build_mask(*it);

				STRONG_ASSERT(vols.size() == 3);

				bvolume_data_ptr xmsk = vols.front(); vols.pop_front();
				bvolume_data_ptr ymsk = vols.front(); vols.pop_front();
				bvolume_data_ptr zmsk = vols.front(); vols.pop_front();

				add_mask(xmsk, vec3i(1,1,0));
				add_mask(ymsk, vec3i(1,1,0));
				add_mask(zmsk, vec3i(0,0,1));
			
				std::cout << std::endl;
			}
		}
	}

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

// sampling
protected:
    virtual sample_ptr get_sample( std::size_t idx )
    {
    	sample_ptr s = volume_data_provider::get_sample(idx);

    	crop_affinity(s->labels);
    	crop_affinity(s->masks);

    	return s;
	}

	template <typename T>
    T crop_affinity( T vol, vec3i sft )
    {
        vec3i sz = size_of(vol);
        STRONG_ASSERT(sz[0] > sft[0]);
        STRONG_ASSERT(sz[1] > sft[1]);
        STRONG_ASSERT(sz[2] > sft[2]);

        return volume_utils::crop(vol, vec3i::zero, sz - sft);
    }

    template <typename T>
    void crop_affinity( std::list<T>& affs )
    {
        STRONG_ASSERT(affs.size() == 3);

        T xaff = affs.front(); affs.pop_front();
        T yaff = affs.front(); affs.pop_front();
        T zaff = affs.front(); affs.pop_front();

        affs.push_back(crop_affinity(xaff, vec3i(1,1,0)));
        affs.push_back(crop_affinity(yaff, vec3i(1,1,0)));
        affs.push_back(crop_affinity(zaff, vec3i(0,0,1)));
    }

protected:
	virtual void add_label( dvolume_data_ptr lbl, vec3i sft = vec3i::zero )
	{
		std::size_t idx = lbls_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		vec3i FoV = out_szs_[idx];

		// for affinity graph transformation
		FoV += sft;
	
		lbl->set_FoV(FoV,sft);
		lbls_.push_back(lbl);
	}

	virtual void add_mask( bvolume_data_ptr msk, vec3i sft = vec3i::zero )
	{
		std::size_t idx = msks_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		vec3i FoV = out_szs_[idx];

		// for affinity graph transformation
		FoV += sft;
	
		msk->set_FoV(FoV,sft);
		msks_.push_back(msk);
	}


// constructor & destructor
public:
	affinity_data_provider( const std::string& fname, 
						    std::vector<vec3i> in_szs,
						    std::vector<vec3i> out_szs,
						    bool mirroring = false )
		: volume_data_provider()
	{
		in_szs_ = in_szs;
		out_szs_ = out_szs;
		set_FoVs();
		load(fname);
		init(mirroring);
	}

	virtual ~affinity_data_provider()
	{}

}; // class affinity_data_provider

typedef boost::shared_ptr<affinity_data_provider> affinity_data_provider_ptr;

}} // namespace zi::znn

#endif // ZNN_AFFINITY_DATA_PROVIDER_HPP_INCLUDED