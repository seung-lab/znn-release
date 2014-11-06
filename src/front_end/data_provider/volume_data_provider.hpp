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

#ifndef ZNN_VOLUME_DATA_PROVIDER_HPP_INCLUDED
#define ZNN_VOLUME_DATA_PROVIDER_HPP_INCLUDED

#include "data_provider.hpp"
#include "../data_spec/volume_data.hpp"
#include "../data_spec/data_builder.hpp"
#include "../data_spec/data_spec_parser.hpp"
#include "transformer/volume_transformer.hpp"

namespace zi {
namespace znn {

class volume_data_provider : virtual public data_provider
{
protected:
	std::list<dvolume_data_ptr>	imgs_;
	std::list<dvolume_data_ptr>	lbls_;	
	std::list<bvolume_data_ptr>	msks_;

	// rebalanced weight masks
	std::list<dvolume_data_ptr>	wmsks_;

	std::vector<vec3i>			in_szs_;
	std::vector<vec3i>			out_szs_;
	std::vector<vec3i>			FoVs_;

	box							range_;
	std::vector<vec3i>         	locs_;

	// status
	bool						initialized_;

	// transformer
	transformer_ptr				trans_;

public:
	bool initialized() const
	{
		return initialized_;
	}


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
			
		// labels (and rebalanced weight masks)
		FOR_EACH( it, parser.label_specs )
		{
			std::cout << "Loading label [" << (*it)->name << "]" << std::endl;			
			std::list<dvolume_data_ptr> vols = data_builder::build_volume(*it);
			FOR_EACH( jt, vols )
			{
				add_label(*jt);
				add_wmask(*jt);
			}
			std::cout << std::endl;
		}

		// masks
		FOR_EACH( it, parser.mask_specs )
		{
			std::cout << "Loading mask [" << (*it)->name << "]" << std::endl;			
			std::list<bvolume_data_ptr> vols = data_builder::build_mask(*it);
			FOR_EACH( jt, vols )
			{
				add_mask(*jt);
			}
			std::cout << std::endl;
		}
	}


// data augmentation
public:
	void data_augmentation( bool data_aug = false )
	{
		if ( data_aug )
		{
			trans_ = transformer_ptr(new volume_transformer());
		}
		else
		{
			trans_.reset();
		}
	}


// input sampling
public:
	// (randomized) sequential sampling
	virtual sample_ptr next_sample()
	{
		return random_sample();
	}

	// random sampling
	virtual sample_ptr random_sample()
	{
		std::size_t e = get_random_index();
        return get_sample(e);
	}

// sampling
protected:
    virtual sample_ptr get_sample( std::size_t idx )
    {
    	vec3i loc = locs_[idx];

    	// images
    	std::list<double3d_ptr> imgs;
    	FOR_EACH( it, imgs_ )
    	{
    		imgs.push_back((*it)->get_patch(loc));
    	}

    	// labels
    	std::list<double3d_ptr> lbls;
    	FOR_EACH( it, lbls_ )
    	{
    		lbls.push_back((*it)->get_patch(loc));
    	}

    	// rebalance masks
    	std::list<double3d_ptr> wmsks;
    	FOR_EACH( it, wmsks_ )
    	{
    		wmsks.push_back((*it)->get_patch(loc));
    	}

    	// masks
    	std::list<bool3d_ptr> msks;
    	FOR_EACH( it, msks_ )
    	{
    		msks.push_back((*it)->get_patch(loc));
    	}

    	sample_ptr s = sample_ptr(new sample(imgs, lbls, wmsks, msks));
    	if ( trans_ ) trans_->transform(s);
    	return s;
    }

// random sampling
protected:
	std::size_t get_random_index()
    {
        return rand() % locs_.size();
    }


protected:
	virtual void add_image( dvolume_data_ptr img )
	{
		std::size_t idx = imgs_.size();
		STRONG_ASSERT(idx < in_szs_.size());		
		
		img->set_FoV(in_szs_[idx]);
		imgs_.push_back(img);
	}

	virtual void add_label( dvolume_data_ptr lbl )
	{
		std::size_t idx = lbls_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		vec3i FoV = out_szs_[idx];
		
		lbl->set_FoV(FoV);
		lbls_.push_back(lbl);
	}

	virtual void add_mask( bvolume_data_ptr msk )
	{
		std::size_t idx = msks_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		vec3i FoV = out_szs_[idx];
		
		msk->set_FoV(FoV);
		msks_.push_back(msk);
	}

	virtual void add_wmask( dvolume_data_ptr lbl )
	{
		std::size_t idx = wmsks_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		vec3i FoV = out_szs_[idx];
		
		double3d_ptr wmsk = 
			volume_utils::binomial_rebalance_mask(lbl->get_volume());

		dvolume_data_ptr vd = 
			dvolume_data_ptr(new dvolume_data(wmsk));
		
		vd->set_offset(lbl->get_offset());
		vd->set_FoV(FoV);
		wmsks_.push_back(vd);
	}

	void update_range()
	{
		range_ = imgs_.front()->get_range();

		FOR_EACH( it, imgs_ )
		{
			range_ = range_.intersect((*it)->get_range());
		}

		FOR_EACH( it, lbls_ )
		{
			range_ = range_.intersect((*it)->get_range());
		}
	}


protected:
    virtual void init()
    {
    	// integrity check
    	STRONG_ASSERT(in_szs_.size()  == imgs_.size());    	
	    STRONG_ASSERT(out_szs_.size() == lbls_.size());
	    STRONG_ASSERT(out_szs_.size() == msks_.size());

	    // boundary mirroring


        // valid locations
        collect_valid_locations();

        initialized_ = true;
    }

    void collect_valid_locations()
	{
		std::cout << "[volume_data_provider] collect_valid_locations" << std::endl;
        zi::wall_timer wt;

        update_range();

        vec3i uc = range_.upper_corner();
        vec3i lc = range_.lower_corner();
        std::cout << "Upper corner: " << uc << std::endl;
        std::cout << "Lower corner: " << lc << std::endl;
        for ( std::size_t z = uc[2]; z < lc[2]; ++z )
            for ( std::size_t y = uc[1]; y < lc[1]; ++y )
                for ( std::size_t x = uc[0]; x < lc[0]; ++x )
                {
                	vec3i loc(x,y,z);
                	bool valid = false;
                	FOR_EACH( it, msks_ )
                	{
                		valid |= (*it)->get_value(loc);
                	}
                	if ( valid ) locs_.push_back(loc);
                }

        std::cout << "Number of valid samples: " << locs_.size() << std::endl;
		std::cout << "Completed. (Elapsed time: " 
                  << wt.elapsed<double>() << " secs)\n" << std::endl;
    }

    void set_FoVs()
    {
		vec3i out_sz = out_szs_.front();

		// [kisuklee]
		// Currently, output sizes are assumed to be the same.
		FOR_EACH( it, out_szs_ )
		{
			STRONG_ASSERT(*it == out_sz);
		}

		FoVs_.clear();
		FOR_EACH( it, in_szs_ )
		{
			FoVs_.push_back(*it - out_sz);
		}
    }


// for debugging
public:
	void print() const
	{
	}

	void save( const std::string& fpath ) const
	{
		// images
		std::size_t cnt = 0;
		FOR_EACH( it, imgs_ )
		{
			std::string idx = boost::lexical_cast<std::string>(cnt++);
			std::string fname = fpath + ".input." + idx;
			double3d_ptr img = (*it)->get_volume();
			volume_utils::save(img, fname);
			export_size_info(size_of(img), fname);
		}
			

		// labels
		cnt = 0;
		FOR_EACH( it, lbls_ )
		{
			std::string idx = boost::lexical_cast<std::string>(cnt++);
			std::string fname = fpath + ".label." + idx;
			double3d_ptr lbl = (*it)->get_volume();
			volume_utils::save(lbl, fname);
			export_size_info(size_of(lbl), fname);
		}

		// masks
		cnt = 0;
		FOR_EACH( it, msks_ )
		{
			std::string idx = boost::lexical_cast<std::string>(cnt++);
			std::string fname = fpath + ".mask." + idx;
			bool3d_ptr msk = (*it)->get_volume();
			volume_utils::save(msk, fname);
			export_size_info(size_of(msk), fname);	
		}
	}


// constructor & destructor
public:
	volume_data_provider()
		: initialized_(false)
		, trans_()
	{}

	volume_data_provider( const std::string& fname, 
						   std::vector<vec3i> in_szs,
						   std::vector<vec3i> out_szs )
		: in_szs_(in_szs)
		, out_szs_(out_szs)
		, initialized_(false)
		, trans_()
	{
		set_FoVs();
		load(fname);
		init();
	}

	virtual ~volume_data_provider()
	{}

}; // class volume_data_provider

typedef boost::shared_ptr<volume_data_provider> volume_data_provider_ptr;

}} // namespace zi::znn

#endif // ZNN_VOLUME_DATA_PROVIDER_HPP_INCLUDED