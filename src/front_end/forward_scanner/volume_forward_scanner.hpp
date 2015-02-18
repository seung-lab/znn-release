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

#ifndef ZNN_VOLUME_FORWARD_SCANNER_HPP_INCLUDED
#define ZNN_VOLUME_FORWARD_SCANNER_HPP_INCLUDED

#include "forward_scanner.hpp"
#include "../data_spec/rw_volume_data.hpp"

#include <string>

namespace zi {
namespace znn {

class volume_forward_scanner : virtual public forward_scanner
{
private:
	typedef std::set<std::size_t> 	scan_coord;

private:
	std::list<dvolume_data_ptr>		imgs_;
	std::list<rw_dvolume_data_ptr>	outs_;

	std::vector<vec3i>				in_szs_ ;
	std::vector<vec3i>				out_szs_;
	std::vector<vec3i>				FoVs_	;

	box								range_;

	vec3i							scan_offset_;
	vec3i							scan_step_	;
	vec3i							scan_dim_	;
	std::vector<scan_coord>			scan_coords_;
	std::set<vec3i>					scan_locs_	;
	vec3i							scan_uc_	;
	vec3i							scan_lc_	;
	std::list<vec3i>				wait_queue_	;


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
		}
	}

	virtual void init( bool mirroring = false )
	{
		// boundary mirroring
	    if ( mirroring ) boundary_mirroring();

		// updating range should precede anything else
		update_range();

		// the setting order below should be strictly followed
		set_scanning_offset();
		set_scanning_dimensions();
		set_scanning_locations();

		std::cout << "[volume_forward_scanner]"			<< '\n'
		  		  << "offset:    " << scan_offset_ 		<< '\n'
		  		  << "step size: " << scan_step_ 		<< '\n'
		  		  << "dimension: " << scan_dim_			<< '\n'
		  		  << "num locs:  " << scan_locs_.size() << std::endl;

		// build output volumes
		prepare_outputs();

		// debug print
		print();
	}

protected:
	// [12/02/2014 kisukee]
    // Implementation is still not stable.
    virtual void boundary_mirroring()
    {
    	std::cout << "[volume_forward_scanner] boundary_mirroring" << std::endl;

    	std::set<std::size_t> x;
    	std::set<std::size_t> y;
    	std::set<std::size_t> z;
    	
    	FOR_EACH( it, FoVs_ )
    	{
    		vec3i FoV = *it;
    		x.insert(FoV[0]/2);
    		y.insert(FoV[1]/2);
    		z.insert(FoV[2]/2);
    	}

    	STRONG_ASSERT(!x.empty());
    	STRONG_ASSERT(!y.empty());
    	STRONG_ASSERT(!z.empty());

    	vec3i offset(*x.rbegin(), *y.rbegin(), *z.rbegin());

    	std::cout << "offset = " << offset << std::endl;

    	std::list<dvolume_data_ptr>	mimgs;
    	std::vector<vec3i>::iterator fov = FoVs_.begin();
    	std::vector<vec3i>::iterator insz = in_szs_.begin();
		FOR_EACH( it, imgs_ )
		{
			dvolume_data_ptr img = *it;

			// field of view
			vec3i FoV = *fov++;
			
			// mirrored volume
			double3d_ptr mimg = 
				volume_utils::mirror_boundary(img->get_volume(),FoV);
			std::cout << "msize = " << size_of(mimg) << std::endl;

			// new data volume
			dvolume_data_ptr 
				vd = dvolume_data_ptr(new dvolume_data(mimg));
				vd->set_offset(img->get_offset() + offset - FoV/vec3i(2,2,2));
				vd->set_FoV(*insz++);

			mimgs.push_back(vd);
		}

		// swap with new mirrored image volumes
		imgs_.swap(mimgs);
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
			vec3i in_sz = *it;
			FoVs_.push_back(in_sz - out_sz + vec3i::one);
		}
    }

private:
	void set_scanning_offset()
	{
		vec3i uc = range_.upper_corner();

		scan_offset_ = uc + scan_offset_;
		
		// sanity check
		STRONG_ASSERT(range_.contains(scan_offset_));
	}

	void set_scanning_dimensions()
	{		
		set_scanning_dimension(0); // x-direction
		set_scanning_dimension(1); // y-direction
		set_scanning_dimension(2); // z-direction
	}

	void set_scanning_dimension( std::size_t dim )
	{
		// 0: x-direction
		// 1: y-direction
		// 2: z-direction
		STRONG_ASSERT(dim < 3);
		STRONG_ASSERT(!out_szs_.empty());

		std::vector<std::size_t> v;
		FOR_EACH( it, out_szs_ )
		{
			v.push_back((*it)[dim]);
		}

		std::vector<std::size_t>::iterator min_it
			= std::min_element(v.begin(),v.end());
		scan_step_[dim] = (*min_it);

		vec3i uc = scan_offset_;
		vec3i lc = range_.lower_corner() - vec3i::one;

		// automated full span scanning
		if ( scan_dim_[dim] == 0 )
		{
			scan_dim_[dim] = (lc[dim]-uc[dim])/scan_step_[dim] + 1;
			STRONG_ASSERT(scan_dim_[dim] > 0);

			// offcut solver
			scan_coords_[dim].insert(lc[dim]);
		}

		// scan coordinates
		std::size_t loc = scan_offset_[dim];
		for ( std::size_t ix = 0; ix < scan_dim_[dim]; ++ix )
		{
			scan_coords_[dim].insert(loc);
			loc += scan_step_[dim];
		}		
		
		// sanity check
		STRONG_ASSERT((uc[dim] + (scan_dim_[dim]-1)*scan_step_[dim]) <= lc[dim]);
	}

	void set_scanning_locations()
	{
		FOR_EACH( x, scan_coords_[0] )
		{
			FOR_EACH( y, scan_coords_[1] )
			{
				FOR_EACH( z, scan_coords_[2] )
				{
					scan_locs_.insert(vec3i(*x,*y,*z));	
				}
			}
		}

		scan_uc_ = *scan_locs_.begin();
		scan_lc_ = *scan_locs_.rbegin();
	}

	void prepare_outputs()
	{
		FOR_EACH( it, out_szs_ )
		{
			vec3i out_sz = *it;
			box a = box::centered_box(scan_uc_,out_sz);
			box b = box::centered_box(scan_lc_,out_sz);
			add_output(a + b);
		}
	}


private:
	void add_image( dvolume_data_ptr img )
	{
		std::size_t idx = imgs_.size();
		STRONG_ASSERT(idx < in_szs_.size());
		img->set_FoV(in_szs_[idx]);
		imgs_.push_back(img);

		std::cout << "<add_image()>" << std::endl;
		img->print();
	}

	void add_output( const box& range )
	{
		std::size_t idx = outs_.size();
		STRONG_ASSERT(idx < out_szs_.size());

		double3d_ptr vol = volume_pool.get_double3d(range.size());		
		vec3i offset = range.upper_corner();
		vec3i FoV = out_szs_[idx];
		rw_dvolume_data_ptr out =
			rw_dvolume_data_ptr(new rw_dvolume_data(vol,FoV,offset));		
		outs_.push_back(out);

		std::cout << "<add_output()>" << std::endl;
		out->print();
	}

	void update_range()
	{
		range_ = imgs_.front()->get_range();

		FOR_EACH( it, imgs_ )
		{
			range_ = range_.intersect((*it)->get_range());
		}

		std::cout << "[volume_forward_scanner]" << std::endl;
		std::cout << "Updated range: " << range_ << std::endl;
	}


public:
	virtual bool pull( std::list<double3d_ptr>& inputs )
	{
		STRONG_ASSERT(wait_queue_.empty());

		bool ret = false;

		if ( scan_locs_.size() > 0 )
		{
			vec3i loc = *scan_locs_.begin();
			scan_locs_.erase(scan_locs_.begin());
			wait_queue_.push_back(loc);
			
			inputs.clear();
			FOR_EACH( it, imgs_ )
			{
				inputs.push_back((*it)->get_patch(loc));
			}

			ret = true;
		}

		return ret;
	}

	virtual void push( std::list<double3d_ptr>& outputs )
	{
		STRONG_ASSERT(wait_queue_.size() == 1);
		STRONG_ASSERT(outs_.size() == outputs.size());

		vec3i loc = wait_queue_.front();		
		wait_queue_.pop_front();

		std::list<rw_dvolume_data_ptr>::iterator oit = outs_.begin();
		FOR_EACH( it, outputs )
		{
			(*oit++)->set_patch(loc,*it);
		}
	}

	virtual void save( const std::string& fpath ) const
	{
		// outputs
		std::size_t cnt = 0;
		FOR_EACH( it, outs_ )
		{
			std::string idx = boost::lexical_cast<std::string>(cnt++);
			std::string fname = fpath + "." + idx;
			double3d_ptr out = (*it)->get_volume();
			volume_utils::save(out,fname);
			export_size_info(size_of(out),fname);
		}
	}


private:
	// for debugging
	void print() const
	{
		std::cout << "[volume_forward_scanner]"			<< '\n'
				  << "offset:    " << scan_offset_ 		<< '\n'
				  << "step size: " << scan_step_ 		<< '\n'
				  << "dimension: " << scan_dim_			<< '\n'
				  << "num locs:  " << scan_locs_.size() << std::endl;
		
		std::size_t i = 0;
		FOR_EACH( it, outs_ )
		{
			std::cout << "\n<output" << ++i << ">\n";
			(*it)->print();
		}
	}


public:	
	volume_forward_scanner( const std::string& load_path,							
						    std::vector<vec3i> in_szs,
						    std::vector<vec3i> out_szs,
						    vec3i off,
						    vec3i dim,
						    bool mirroring = false )
		: in_szs_(in_szs)
		, out_szs_(out_szs)
		, scan_offset_(off)
		, scan_dim_(dim)
		, scan_coords_(3)
		, scan_locs_()
	{
		set_FoVs();
		load(load_path);
		init(mirroring);
	}

	virtual ~volume_forward_scanner()
	{}

}; // abstract class volume_forward_scanner

typedef boost::shared_ptr<volume_forward_scanner> volume_forward_scanner_ptr;

}} // namespace zi::znn

#endif // ZNN_VOLUME_FORWARD_SCANNER_HPP_INCLUDED