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

#ifndef ZNN_NODE_FACTORY_HPP_INCLUDED
#define ZNN_NODE_FACTORY_HPP_INCLUDED

#include "node_group.hpp"
#include "../../error_fn/error_fns.hpp"
#include "../../initializer/initializers.hpp"

#include <boost/lexical_cast.hpp>
#include <zi/utility/singleton.hpp>

namespace zi {
namespace znn {

class node_factory_impl
{
public: 
	std::map<std::string,node_group_ptr> 	node_group_pool;	 
	std::list<node_ptr> 					node_pool;

public:
	node_group_ptr create_node_group( const std::string& name,
									  node_spec_ptr spec,
									  const std::string& path = "" )
	{
		node_group_ptr g = node_group_ptr(new node_group(name));

		// load if possible
		if ( spec->load && !path.empty() && g->load_spec(path) )
		{
			build_node_group(g);
			g->load_bias(path);
		}
		else
		{
			g->set_spec(spec);
			build_node_group(g);
		}

		register_node_group(g);

		return g;
	}

	node_ptr create_node( const std::string& name, 
						  node_spec_ptr spec,
						  std::size_t neuron_no )
	{
		std::string node_name = name + ":" + 
			boost::lexical_cast<std::string>(neuron_no);		

		error_fn_ptr act_fn = 
			create_activation_function(spec->activation,spec->act_params);

		node_ptr ret = 
			node_ptr(new node(node_name,spec->bias,spec->eta,
							  0,	// dummy layer number
							  neuron_no,act_fn,spec->mom,spec->wc));

		// set filtering - set stride later		
		set_filter(ret,spec->filter,spec->filter_size);
			
		return ret;
	}

private:
	// [02/06/2014 kisuklee]
	// never, ever call this function more than once!
	void build_node_group( node_group_ptr g )
	{
		for ( std::size_t i = g->count(); i < g->spec()->size; ++i )
		{			
			g->add_node(create_node(g->name(),g->spec(),i+1));			
		}
	}

	void register_node_group( node_group_ptr g )
	{
		node_group_pool[g->name()] = g;
		FOR_EACH( it, g->nodes_ )
		{
			node_pool.push_back(*it);
		}
	}

private:
	error_fn_ptr create_activation_function( const std::string& type,
											 const std::string& params )
	{
		error_fn_ptr ret;		

		// parser for parsing arguments
		std::vector<double> args;
		zi::zargs_::parser<std::vector<double> > arg_parser;
		bool parsed = arg_parser.parse(&args,params);

		if ( type == "logistic" )
		{
			ret = error_fn_ptr(new logistic_error_fn);
		}
		else if ( type == "forward_logistic" )
		{
			ret = error_fn_ptr(new forward_logistic_error_fn);
		}
		else if ( type == "tanh" )
		{
			if ( parsed && (args.size() == 2) )
			{
				ret = error_fn_ptr(new hyperbolic_tangent_error_fn(args[0],args[1]));
			}
			else
			{
				ret = error_fn_ptr(new hyperbolic_tangent_error_fn);
			}
		}
		else if ( type == "softsign" )
		{
			ret = error_fn_ptr(new soft_sign_error_fn);
		}
		else if ( type == "relu" )
		{
			ret = error_fn_ptr(new rectify_linear_error_fn);
		}
		else if ( type == "linear" )
		{
			if ( parsed && (args.size() == 2) )
			{
				ret = error_fn_ptr(new linear_error_fn(args[0],args[1]));
			}
			else
			{
				ret = error_fn_ptr(new linear_error_fn);
			}
		}
		else
		{
			std::cout << "Unknown activation function type: " << type << std::endl;
			STRONG_ASSERT(false);
		}

		return ret;
	}

	void set_filter( node_ptr a, const std::string& type, vec3i filter_size )
	{
		if ( type == "max" )
		{
			a->set_filtering(std::greater<double>(),filter_size);
		}
		else
		{
			std::cout << "Unknown filter type: " << type << std::endl;
			STRONG_ASSERT(false);	
		}

	}


public:
	void initialize_bias( node_group_ptr g )
	{
		node_spec_ptr 	spec = g->spec();
		initializer_ptr init = create_initializer(g);

		std::size_t n = spec->size;
		vec3i sz(n,1,1);
		double3d_ptr w = volume_pool.get_double3d(sz);
		volume_utils::zero_out(w);
		init->initialize(w);

		for ( std::size_t i = 0; i < n; ++i )
		{
			g->nodes_[i]->set_bias(w->data()[i]);
		}
	}

private:
	initializer_ptr create_initializer( node_group_ptr g )
	{
		node_spec_ptr 	spec = g->spec();
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
		else if ( spec->init_type == "const" )
		{
			ret = initializer_ptr(new Constant_init(spec->bias));
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
	node_factory_impl()
		: node_group_pool()
		, node_pool()
	{}

}; // class node_factory

namespace {
node_factory_impl& node_factory =
	zi::singleton<node_factory_impl>::instance();
} // anonymous namespace

}} // namespace zi::znn

#endif // ZNN_NODE_FACTORY_HPP_INCLUDED
