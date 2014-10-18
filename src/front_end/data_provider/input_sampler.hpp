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

#ifndef ZNN_INPUT_SAMPLER_HPP_INCLUDED
#define ZNN_INPUT_SAMPLER_HPP_INCLUDED

#include "../../core/types.hpp"
#include "../../core/volume_utils.hpp"

#include <cstdlib>

namespace zi {
namespace znn {

class input_sampler
{
private:
	std::list<double3d_ptr>		inputs_;
	std::list<double3d_ptr> 	labels_;	// template labels
	std::list<bool3d_ptr>		masks_ ;	// template masks
	
	bool affin_;
	std::list<double3d_ptr> 	ret_lbls_;	// labels for actual return
	std::list<bool3d_ptr>   	ret_msks_;	// masks for actual return


public:
	std::list<double3d_ptr> get_inputs()
	{
		return inputs_;
	}

	std::list<double3d_ptr> get_labels()
	{
		return ret_lbls_;
	}

	std::list<bool3d_ptr> get_masks()
	{
		return ret_msks_;
	}


public:
	void save( const std::string& fpath )
	{
		// inputs
		volume_utils::save_list(inputs_, fpath + ".input");
		vec3i sz = size_of(inputs_.front());
		export_size_info(sz, fpath + ".input");

		// labels
		volume_utils::save_list(ret_lbls_, fpath + ".label");
		sz = size_of(labels_.front());
		export_size_info(sz, fpath + ".label");
	}


public:
	input_sampler( std::list<double3d_ptr> i, 
				   std::list<double3d_ptr> l, 
				   std::list<bool3d_ptr> m,
				   bool affin = false )
		: inputs_(i)
		, labels_(l)
		, masks_(m)
		, affin_(affin)
		, ret_lbls_(l)
		, ret_msks_(m)
	{	
		init();
	}

private:
	void init()
	{
		// affinity graph
		if( affin_ )
		{
			update_return_labels();
		}
	}

public:
	std::pair<bool,std::size_t> random_transform()
	{
		bool tp = (rand() % 2 == 1 ? true : false);
		std::size_t flip = rand() % 8;

		transform(tp,flip);

		return std::make_pair(tp,flip);
	}

	// [12/03/2013 kisuklee]
	// test correctness
	void transform( bool tp, std::size_t flip )
	{
		// transpose
		if ( tp ) transpose();

		// flip
		flipdim(flip % 8);

		update_return_labels();
	}

private:
	void flipdim( std::size_t dim )
	{
		// inputs
		FOR_EACH( it, inputs_ )
		{
			(*it) = volume_utils::flipdim(*it, dim);
		}

		// labels
        FOR_EACH( it, labels_ )
        {
            (*it) = volume_utils::flipdim(*it, dim);
        }

        // masks
        FOR_EACH( it, masks_ )
        {
        	(*it) = volume_utils::flipdim(*it, dim);
        }
	}

	void transpose()
	{
		// inputs
		FOR_EACH( it, inputs_ )
		{
			(*it) = volume_utils::transpose(*it);
		}		

		// labels
        FOR_EACH( it, labels_ )
        {
            (*it) = volume_utils::transpose(*it);
        }
        if ( affin_ )
        {
        	STRONG_ASSERT(labels_.size() >= 2);
        	std::list<double3d_ptr>::iterator xit = labels_.begin();
        	std::list<double3d_ptr>::iterator yit = labels_.begin();
        	++yit; (*xit).swap(*yit);
        }

        // masks
        FOR_EACH( it, masks_ )
        {
        	(*it) = volume_utils::transpose(*it);
        }
	}

	void update_return_labels()
	{
		if( affin_ )
		{
			// labels
			ret_lbls_.clear();			
			FOR_EACH( it, labels_ )
			{
				vec3i s = size_of(*it);
				ret_lbls_.push_back(volume_utils::crop_left((*it),s-vec3i::one));
			}

			// masks
			ret_msks_.clear();			
			FOR_EACH( it, masks_ )
			{
				vec3i s = size_of(*it) - vec3i::one;				
				ret_msks_.push_back(volume_utils::crop((*it),s[0],s[1],s[2]));
			}
		}
		else
		{
			ret_lbls_ = labels_;
			ret_msks_ = masks_;
		}
	}

}; // class input_sampler

typedef boost::shared_ptr<input_sampler> input_sampler_ptr;

}} // namespace zi::znn

#endif // ZNN_INPUT_SAMPLER_HPP_INCLUDED