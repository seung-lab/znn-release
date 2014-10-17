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
	std::list<dvolume_data_ptr>		imgs_;
	std::list<rw_dvolume_data_ptr>	outs_;

	std::vector<vec3i>				in_szs_;
	std::vector<vec3i>				out_szs_;

	box								range_;

	vec3i							scan_offset_;
	vec3i							scan_step_;
	vec3i							scan_dim_;
	std::list<vec3i>				scan_locs_;
	std::list<vec3i>				wait_queue_;


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

	virtual void init()
	{
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

		if ( scan_dim_[dim] == 0 )
		{
			scan_dim_[dim] = (lc[dim]-uc[dim])/scan_step_[dim] + 1;
			STRONG_ASSERT(scan_dim_[dim] > 0);
		}
		
		// sanity check
		STRONG_ASSERT((uc[dim] + (scan_dim_[dim]-1)*scan_step_[dim]) <= lc[dim]);
	}

	void set_scanning_locations()
	{
		std::size_t bx = scan_offset_[0];
		std::size_t by = scan_offset_[1];
		std::size_t bz = scan_offset_[2];		

		std::size_t sx = scan_step_[0];
		std::size_t sy = scan_step_[1];
		std::size_t sz = scan_step_[2];

		std::size_t x = bx;
		std::size_t y = by;
		std::size_t z = bz;

		for ( std::size_t ix = 0; ix < scan_dim_[0]; ++ix, x += sx, y = by )
			for ( std::size_t iy = 0; iy < scan_dim_[1]; ++iy, y += sy, z = bz )
				for ( std::size_t iz = 0; iz < scan_dim_[2]; ++iz, z += sz )
				{
					scan_locs_.push_back(vec3i(x,y,z));
				}
	}

	void prepare_outputs()
	{
		// vec3i uc = scan_locs_.front();
		// vec3i lc = scan_locs_.back();
		vec3i uc = scan_offset_;
		vec3i lc = uc + (scan_dim_ - vec3i::one)*scan_step_;

		// [08/19/2014] temporary sanity check
		STRONG_ASSERT(scan_locs_.front() == uc);
		std::cout << scan_locs_.back() << std::endl;
		std::cout << lc << std::endl;
		STRONG_ASSERT(scan_locs_.back() == lc);

		FOR_EACH( it, out_szs_ )
		{
			vec3i out_sz = *it;
			box a = box::centered_box(uc,out_sz);
			box b = box::centered_box(lc,out_sz);
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
			vec3i loc = scan_locs_.front();
			scan_locs_.pop_front();
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
						    vec3i dim )
		: in_szs_(in_szs)
		, out_szs_(out_szs)
		, scan_offset_(off)
		, scan_dim_(dim)
	{
		load(load_path);
		init();
	}

	virtual ~volume_forward_scanner()
	{}

}; // abstract class volume_forward_scanner

typedef boost::shared_ptr<volume_forward_scanner> volume_forward_scanner_ptr;

}} // namespace zi::znn

#endif // ZNN_VOLUME_FORWARD_SCANNER_HPP_INCLUDED