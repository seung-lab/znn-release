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

#ifndef ZNN_NET_BUILDER_HPP_INCLUDED
#define ZNN_NET_BUILDER_HPP_INCLUDED

#include "node_spec/node_factory.hpp"
#include "edge_spec/edge_factory.hpp"
#include "net.hpp"

#include <boost/regex.hpp>

namespace zi {
namespace znn {

class net_builder
{
private:
	const std::string		config_;

	// status
	bool					operable_;

	// map: name -> spec 
	std::map<const std::string,node_spec_ptr> node_specs;
	std::map<const std::string,edge_spec_ptr> edge_specs;


public:
	bool operable() const
	{
		return operable_;
	}

	net_ptr build( const std::string& path = "" )
	{
		net_ptr ret = net_ptr(new net);
		ret->set_load_path(path);

		// construct node groups
		FOR_EACH( it, node_specs )
		{
			const std::string& 	name = it->first;
			node_spec_ptr 		spec = it->second;

			node_group_ptr g = node_factory.create_node_group(name,spec,path);

			// add the newly created node group
			ret->add_node_group(g);
		}

		// construct edge groups
		FOR_EACH( it, edge_specs )
		{
			const std::string& 	name = it->first;
			edge_spec_ptr 		spec = it->second;

			edge_group_ptr g = edge_factory.create_edge_group(name,spec,path);

			// add the newly created edge group
			ret->add_edge_group(g);
		}

		// register inputs/outputs
		ret->find_inputs();
		ret->find_outputs();
		ret->check_dangling();

		return ret;
	}


private:
	bool parse_config()
	{
		std::ifstream fin(config_.c_str());
		if ( !fin )	return false;

		// stringfy
		std::string str( (std::istreambuf_iterator<char>(fin)),
					     (std::istreambuf_iterator<char>()) );
		fin.close();

		// extract node/edge group name list		
		boost::regex re("\\[(.*?)\\]");
		boost::sregex_iterator it(str.begin(),str.end(),re);
		boost::sregex_iterator jt;
		while ( it != jt )
		{
			std::string name = strip_brackets((*it++).str());
			if ( name.empty() )
			{
				std::cout << "[net_builder] parse_config" << std::endl;
				std::string what = "Empty name is not allowed";
				throw std::invalid_argument(what);
			}

			if ( std::string::npos != name.find('_') ) // edge specs
			{
				std::vector<std::string> node_names;
				boost::split(node_names,name,boost::is_any_of("_"));
				if ( node_names[0].empty() || node_names[1].empty() )
				{
					std::cout << "[net_builder] parse_config" << std::endl;
					std::string what = "Invalid edge group name [" + name + "]";
					throw std::invalid_argument(what);
				}

				edge_spec_ptr spec = edge_spec_ptr(new edge_spec(name));
				STRONG_ASSERT( spec->build(config_) );
				edge_specs[name] = spec;
			}
			else // node specs
			{
				node_spec_ptr spec = node_spec_ptr(new node_spec(name));
				STRONG_ASSERT( spec->build(config_) );
				node_specs[name] = spec;
			}
		}

		return true;
	}


public:
	net_builder( const std::string& config )
		: config_(config)
		, operable_(false)
		, node_specs()
		, edge_specs()
	{
		std::cout << "\n";
		operable_ = parse_config();
		std::cout << "[net_builder] operable: " << operable() << std::endl;
		if ( !operable() )
		{
			std::string what 
				= "Failed to build net from the file [" + config + "]";
			throw std::invalid_argument(what);
		}
	}

}; // class net_builder

}} // namespace zi::znn

#endif // ZNN_NET_BUILDER_HPP_INCLUDED
