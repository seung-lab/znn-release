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

#ifndef ZNN_AFFINITY_MASK_HPP_INCLUDED
#define ZNN_AFFINITY_MASK_HPP_INCLUDED

#include "../../core/types.hpp"
#include "../../core/volume_pool.hpp"
#include "../../core/volume_utils.hpp"
#include "../../core/utils.hpp"

namespace zi {
namespace znn {

class affinity_mask;

typedef boost::shared_ptr<affinity_mask> 	affinity_mask_ptr;
typedef std::list<bool3d_ptr> 				  bool3d_ptr_list;

class affinity_mask
{
private:
	std::size_t 		dim_ ;
	bool3d_ptr_list		mask_;
	vec3i				size_;


// constructing affinity mask internally
private:
	void load( const std::string& fname )
	{
		std::string smsk = fname;

        // size of the whole volume
    	vec3i msksz = import_size_info(fname);
    	STRONG_ASSERT( msksz > vec3i::zero );
    	std::cout << "Mask size:\t" << msksz << std::endl;
    	size_ = msksz - vec3i::one;

        // prepare volume for whole mask
        bool3d_ptr msk = volume_pool.get_bool3d(msksz);

        // load each mask
        if ( volume_utils::load( msk, smsk ) )
    	{
    		// generate affinity mask from mask
    		construct_from_mask(msk);
    	}
    	else
    	{
    		std::cout << "Affinity mask size:\t" << size_ << std::endl;
    		
    		// default mask (all true)
    		for ( std::size_t i = 0; i < dim_; ++i )
			{
				bool3d_ptr affin = volume_pool.get_bool3d(size_);
				volume_utils::fill_one(affin);
				mask_.push_back(affin);
			}
    	}
	}

	void construct_from_mask( bool3d_ptr msk )
	{
		zi::wall_timer wt;
		std::cout << "Constructing affinity mask from mask..." << std::endl;

		vec3i msksz = size_of(msk);
		size_ = msksz - vec3i::one;
		std::cout << "Mask size:\t\t" << msksz << std::endl;
		std::cout << "Affinity mask size:\t" << size_ << std::endl;

		for ( std::size_t i = 0; i < dim_; ++i )
		{
			bool3d_ptr affin = volume_pool.get_bool3d(size_);
			volume_utils::zero_out(affin);
			mask_.push_back(affin);
		}

		for ( std::size_t z = 0; z < size_[2]; ++z )
            for ( std::size_t y = 0; y < size_[1]; ++y )
                for ( std::size_t x = 0; x < size_[0]; ++x )
                {                	
                	std::list<bool> v1_list;
                	v1_list.push_back((*msk)[x][y+1][z+1]);	// x-affinity
                	v1_list.push_back((*msk)[x+1][y][z+1]);	// y-affinity
                	v1_list.push_back((*msk)[x+1][y+1][z]);	// z-affinity
                	std::list<bool>::iterator v1lit = v1_list.begin();

                	bool v2 = (*msk)[x+1][y+1][z+1];
                	
                	FOR_EACH( mit, mask_ )
                	{
                		bool3d_ptr affin = *mit;
                		bool v1 = *v1lit;
                		if ( v1 && v2 )
	                		(*affin)[x][y][z] = true;
	                	else
	                		(*affin)[x][y][z] = false;
	                	++v1lit;
                	}
                }

		std::cout << "Completed. (Elapsed time: " 
                  << wt.elapsed<double>() << " secs)\n" << std::endl;
	}


public:
	bool3d_ptr_list get_masks()
	{
		return mask_;
	}

	affinity_mask_ptr get_submask( const vec3i& off, const vec3i& sz )
	{
		// extract submasks
		bool3d_ptr_list submasks;
		FOR_EACH( it, mask_ )
		{
			submasks.push_back(volume_utils::crop((*it), off, sz));
		}
		return affinity_mask_ptr(new affinity_mask(submasks));
	}

	vec3i get_size() const
	{
		return size_;
	}


// Comparison
public:
	inline bool 
	operator==( const affinity_mask& rhs )
	{
		if ( mask_ == rhs.mask_ )
			return true;

		bool3d_ptr_list::const_iterator rit = rhs.mask_.begin();
		FOR_EACH( it, mask_ )
		{
			bool equal = (**it) == (**rit);
			if ( !equal )
				return false;
			++rit;
		}

		return true;
	}


public:
	void save( const std::string& fname )
    {
    	zi::wall_timer wt;
    	std::cout << "<<<   affinity_mask::save   >>>" << std::endl;

        int i = 0;
        FOR_EACH( it, mask_ )
        {
        	std::cout << "Now processing " << i << "th mask..." << std::endl;
            std::ostringstream subname;
            subname << fname << "." << i++;

            volume_utils::save((*it), subname.str());
        }

        std::string ssz = fname;
        export_size_info( get_size(), ssz );

        std::cout << "Completed. (Elapsed time: " 
                  << wt.elapsed<double>() << " secs)\n" << std::endl;
    }


public:
	affinity_mask( const std::string& fname, std::size_t dim = 3 )
		: dim_(dim)	
	{
		load(fname);
	}

	affinity_mask( bool3d_ptr msk, std::size_t dim = 3 )
		: dim_(dim)
	{
		construct_from_mask(msk);
	}

	affinity_mask( const bool3d_ptr_list& mask )
	{
		ZI_ASSERT( mask.size() > 0 );

		mask_ = mask;
		dim_ = mask_.size();		
		size_ = size_of(mask_.front());
	}

}; // class affinity_mask

}} // namespace zi::znn

#endif // ZNN_AFFINITY_MASK_HPP_INCLUDED