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

#ifndef ZNN_EDGE_FACTORY_HPP_INCLUDED
#define ZNN_EDGE_FACTORY_HPP_INCLUDED

#include "edge_group.hpp"
#include "../node_spec/node_factory.hpp"
#include "../../initializer/initializers.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/dynamic_bitset.hpp>
#include <zi/utility/singleton.hpp>


namespace zi {
namespace znn {

class edge_factory_impl
{
public:
	std::map<std::string,edge_group_ptr> 	edge_group_pool;
	std::list<edge_ptr> 					edge_pool;


public: 
	edge_group_ptr create_edge_group( const std::string& name,
									  edge_spec_ptr spec,
									  const std::string& path = "" )
	{
		std::vector<std::string> node_names;
		boost::split(node_names,name,boost::is_any_of("_"));

		if ( node_factory.node_group_pool.find(node_names[0]) ==
			 node_factory.node_group_pool.end() )
		{
			std::cout << "[edge_factory] create_edge_group" << std::endl;
			std::string what = "Non-existent source node group [" + name + "]";
			throw std::invalid_argument(what);
		}

		if ( node_factory.node_group_pool.find(node_names[1]) ==
			 node_factory.node_group_pool.end() )
		{
			std::cout << "[edge_factory] create_edge_group" << std::endl;
			std::string what = "Non-existent target node group [" + name + "]";
			throw std::invalid_argument(what);	
		}

		node_group_ptr source = node_factory.node_group_pool[node_names[0]];
		node_group_ptr target = node_factory.node_group_pool[node_names[1]];

		return create_edge_group(spec,source,target,path);
	}

	edge_group_ptr create_edge_group( edge_spec_ptr spec,
									  node_group_ptr source,
									  node_group_ptr target,
									  const std::string& path = "" )
	{
		STRONG_ASSERT(spec);
		STRONG_ASSERT(source);
		STRONG_ASSERT(target);

		const std::string& name = source->name() + "_" + target->name();
		edge_group_ptr g = 
			edge_group_ptr(new edge_group(name,source,target));
		
		// load if possible
		if ( spec->load && !path.empty() && g->load_spec(path) )
		{
			build_edge_group(source,target,g);
			// std::cout << "Weight loaded: " << g->load_weight(path) << std::endl;
			g->load_weight(path);
		}
		else
		{
			g->set_spec(spec);
			build_edge_group(source,target,g);
		}

		register_edge_group(g);	
		return g;
	}

private:
	// [02/06/2014 kisuklee]
	// never, ever call this function more than once!
	void build_edge_group( node_group_ptr source,
						   node_group_ptr target,
						   edge_group_ptr g )
	{		
		FOR_EACH( it, source->nodes_ )
		{
			FOR_EACH( jt, target->nodes_ )
			{
				g->add_edge(create_edge(*it,*jt,g));
			}
		}

		source->add_out_connection(g);
		target->add_in_connection(g);
	}

	edge_ptr create_edge( node_ptr source, 
						  node_ptr target, 
						  edge_group_ptr g )
	{		
		double3d_ptr w = create_weight(g);

		edge_spec_ptr spec = g->spec();
		edge_ptr e = edge_ptr(new edge(source,target,w,
									   spec->eta,spec->mom,spec->wc));

		source->add_out_edge(e);
		target->add_in_edge(e);

		edge_pool.push_back(e);
		return e;
	}

	double3d_ptr create_weight( edge_group_ptr g )
	{
		edge_spec_ptr spec = g->spec();
		double3d_ptr w = volume_pool.get_double3d(spec->size);
		volume_utils::zero_out(w);
		return w;
	}

	void register_edge_group( edge_group_ptr g )
	{
		edge_group_pool[g->name()] = g;
		FOR_EACH( it, g->edges_ )
		{
			edge_pool.push_back(*it);
		}
	}


public:
	void initialize_weight( edge_group_ptr g )
	{
		edge_spec_ptr 	spec = g->spec();
		initializer_ptr init = create_initializer(g);

		FOR_EACH( it, g->edges_ )
		{
			double3d_ptr w = volume_pool.get_double3d(spec->size);
			volume_utils::zero_out(w);
			init->initialize(w);
			(*it)->reset_W(w);
		}
	}

private:
	initializer_ptr create_initializer( edge_group_ptr g )
	{
		edge_spec_ptr 	spec = g->spec();
		initializer_ptr ret;
		
		if ( spec->init_type == "Gaussian" )
		{
			ret = initializer_ptr(new Gaussian_init);
			ret->init(spec->init_params);
		}
		else if ( spec->init_type == "Uniform" )
		{
			ret = initializer_ptr(new Uniform_init);
			ret->init(spec->init_params);
		}
		else if ( spec->init_type == "standard" )
		{
			double m = g->target_->fan_in();
			double r = 1/std::sqrt(m);
			ret = initializer_ptr(new Uniform_init(-r,r));
		}
		else if ( spec->init_type == "normalized" )
		{
			double m1 = g->target_->fan_in();
			double m2 = g->target_->fan_out();
			double r  = std::sqrt(6)/std::sqrt(m1+m2);
			ret = initializer_ptr(new Uniform_init(-r,r));
		}
		else if ( spec->init_type == "zero" )
		{
			ret = initializer_ptr(new Zero_init);
		}
		else
		{
			std::string what = "Unknown initializer ["+ spec->init_type + "]";
			throw std::invalid_argument(what);
		}		
		
		return ret;
	}


public:
	edge_factory_impl()
		: edge_group_pool()
		, edge_pool()
	{}

}; // class edge_factory

namespace {
edge_factory_impl& edge_factory =
	zi::singleton<edge_factory_impl>::instance();
} // anonymous namespace

}} // namespace zi::znn

#endif // ZNN_EDGE_FACTORY_HPP_INCLUDED