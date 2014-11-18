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

#ifndef ZNN_DATA_BUILDER_HPP_INCLUDED
#define ZNN_DATA_BUILDER_HPP_INCLUDED

#include "data_spec.hpp"
#include "../../initializer/initializers.hpp"
#include "affinity_graph.hpp"
#include "affinity_mask.hpp"
#include "volume_data.hpp"


namespace zi {
namespace znn {

class data_builder
{
public:
	static
	std::list<dvolume_data_ptr> build_volume( data_spec_ptr spec )
	{
		// load volume
		double3d_ptr vol = volume_pool.get_double3d(spec->size);
		if ( !spec->get_path().empty() )
		{
			if ( !volume_utils::load(vol,spec->get_path()) )
			{
				std::string what = "Failed to load volume [" + spec->get_path() + "]";
				throw std::invalid_argument(what);
			}
		}

		// double volume list
		return preprocess(vol,spec);
	}

	static
	std::list<bvolume_data_ptr> build_mask( data_spec_ptr spec )
	{		
		// load volume
		bool3d_ptr vol = volume_pool.get_bool3d(spec->size);
		if ( !spec->get_path().empty() )
		{
			if ( !volume_utils::load(vol,spec->get_path()) )
			{
				std::string what = "Failed to load volume [" + spec->get_path() + "]";
				throw std::invalid_argument(what);
			}
		}

		// bool volume list
		return preprocess(vol,spec);
	}


private:
	static std::list<dvolume_data_ptr> 
	preprocess( double3d_ptr vol, data_spec_ptr spec )
	{
		std::list<double3d_ptr> vols;
		vols.push_back(vol);

		if ( !spec->pptype.empty() )
		{
			zi::wall_timer wt;
			if ( spec->pptype == "affinity" )
			{
				vols.clear();
				vols = get_affinity_graph(vol,spec->ppargs);
			}
			else if ( spec->pptype == "binary_class_affinity" )
			{
				vols.clear();
				vols = get_binary_class_affinity(vol);
			}
			else if ( spec->pptype == "standard2D" )
			{
				volume_utils::normalize_volume_for_2D(vol);
			}
			else if ( spec->pptype == "standard3D" )
			{
				volume_utils::normalize_volume(vol);
			}
			else if ( spec->pptype == "binarize" )
			{
				volume_utils::binarize(vol);
			}
			else if ( spec->pptype == "binary_class" )
			{
				vols.clear();
				volume_utils::binarize(vol);
				vols = volume_utils::encode_multiclass(vol,2);
			}
			else if ( spec->pptype == "multiclass" )
			{
				vols.clear();
				vols = encode_multiclass(vol,spec->ppargs);
			}
			else if ( spec->pptype == "zero" )
			{
				volume_utils::zero_out(vol);
			}
			else if ( spec->pptype == "one" )
			{
				volume_utils::fill_one(vol);
			}
			else if ( spec->pptype == "const" )
			{
				initializer_ptr init = initializer_ptr(new Constant_init);
				init->init(spec->ppargs);
				init->initialize(vol);
			}
			else if ( spec->pptype == "Uniform" )
			{
				initializer_ptr init = initializer_ptr(new Uniform_init);
				init->init(spec->ppargs);
				init->initialize(vol);
			}
			else if ( spec->pptype == "Gaussian" )
			{
				initializer_ptr init = initializer_ptr(new Gaussian_init);
				init->init(spec->ppargs);
				init->initialize(vol);
			}
			else if ( spec->pptype == "transform" )
			{
				initializer_ptr init = initializer_ptr(new Transform_init);
				init->init(spec->ppargs);
				init->initialize(vol);
			}
			else
			{
				std::cout << "Skip unknown preprocessing [" << spec->pptype 
						  << "]" << std::endl;
			}

			std::cout << "Preprocessing [" << spec->pptype << "] "
					  << "completed. (Elapsed time: " 
	                  << wt.elapsed<double>() << " secs)" << std::endl;
		}

		std::list<dvolume_data_ptr> ret;

		FOR_EACH( it, vols )
		{
			// double volume data
			dvolume_data_ptr 
				vd = dvolume_data_ptr(new dvolume_data(*it));
				vd->set_offset(spec->offset);

			ret.push_back(vd);
		}

		return ret;
	}

	static std::list<bvolume_data_ptr> 
	preprocess( bool3d_ptr vol, data_spec_ptr spec )
	{
		std::list<bool3d_ptr> vols;
		vols.push_back(vol);

		if ( spec->pptype == "affinity" )
		{
			vols.clear();
			vols = get_affinity_mask(spec);
		}
		else
		{
			if ( spec->pptype == "zero" )
			{
				volume_utils::zero_out(vol);
			}	
			else if ( spec->pptype == "one" )
			{
				volume_utils::fill_one(vol);	
			}
			else
			{
				std::string what = 
					"Unknown preprocess type [" + spec->pptype + "]";
				throw std::invalid_argument(what);
			}

			// parser for parsing preprocessing parameters
			std::vector<double> args;
			zi::zargs_::parser<std::vector<double> > arg_parser;
			bool parsed = arg_parser.parse(&args,spec->ppargs);
			
			std::size_t nmask = 1;
			if ( parsed && (args.size() >= 1) )
	        {
	        	nmask = args[0];
	        }

	        for ( std::size_t i = 1; i < nmask; ++i )
				vols.push_back(vol);
		}

		std::list<bvolume_data_ptr> ret;

		FOR_EACH( it, vols )
		{
			// bool volume data
			bvolume_data_ptr 
				vd = bvolume_data_ptr(new bvolume_data(*it));
				vd->set_offset(spec->offset);

			ret.push_back(vd);
		}

		return ret;
	}

	static std::list<double3d_ptr> 
	get_affinity_graph( double3d_ptr vol, const std::string& params )
	{
		// parser for parsing preprocessing parameters
		std::vector<double> args;
		zi::zargs_::parser<std::vector<double> > arg_parser;
		bool parsed = arg_parser.parse(&args,params);
		
		std::size_t dim = 3;
		double neg = static_cast<double>(0);
		double pos = static_cast<double>(1);
		if ( parsed && (args.size() >= 2) )
		{
			neg = args[0];
			pos = args[1];
		}
		
		affinity_graph_ptr affin = 
			affinity_graph_ptr(new affinity_graph(vol,dim,pos,neg));

		return affin->get_labels();
	}

	static std::list<double3d_ptr> 
	get_binary_class_affinity( double3d_ptr vol )
	{
		// 0,1 affinity graph
		std::list<double3d_ptr> affin = get_affinity_graph(vol,"");

		std::list<double3d_ptr> ret;
		FOR_EACH( it, affin )
		{
			std::list<double3d_ptr> bin = 
				volume_utils::encode_multiclass(*it,2);				
			ret.insert(ret.end(),bin.begin(),bin.end());
		}

		return ret;
	}

	static std::list<double3d_ptr> 
	encode_multiclass( double3d_ptr vol, const std::string& params )
	{
		// parser for parsing preprocessing parameters
		std::vector<double> args;
		zi::zargs_::parser<std::vector<double> > arg_parser;
		bool parsed = arg_parser.parse(&args,params);
		
		std::size_t nclass = 2;
        if ( parsed && (args.size() >= 1) )
        {
        	nclass = args[0];
        }
        
		return volume_utils::encode_multiclass(vol,nclass);
	}

	// [07/30/2014]
	// Clumsy implementation. Should modify later.
	static std::list<bool3d_ptr> get_affinity_mask( data_spec_ptr spec )
	{
		std::size_t dim = 3;

		std::list<bool3d_ptr> ret;
		if ( !spec->get_path().empty() )
		{			
			std::cout << "Affinity mask path: " << spec->get_path() << std::endl;

			bool3d_ptr vol = volume_pool.get_bool3d(spec->size);
			volume_utils::load(vol,spec->get_path());
			affinity_mask_ptr msk = 
				affinity_mask_ptr(new affinity_mask(vol));
			ret = msk->get_masks();
		}
		else
		{
			for ( std::size_t i = 0; i < dim; ++i )
			{
				bool3d_ptr vol = volume_pool.get_bool3d(spec->size);
				volume_utils::fill_one(vol);
				ret.push_back(vol);
			}
		}

		return ret;
	}

}; // class data_builder

typedef boost::shared_ptr<data_builder> data_builder_ptr;

}} // namespace zi::znn

#endif // ZNN_DATA_BUILDER_HPP_INCLUDED