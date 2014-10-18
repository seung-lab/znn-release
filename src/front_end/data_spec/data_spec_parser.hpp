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

#ifndef ZNN_DATA_SPEC_PARSER_HPP_INCLUDED
#define ZNN_DATA_SPEC_PARSER_HPP_INCLUDED

#include "data_spec.hpp"

#include <boost/regex.hpp>


namespace zi {
namespace znn {

class data_spec_parser
{
private:
	const std::string		config_;


public:
	// map: name -> spec
	// std::map<const std::string,data_spec_ptr> 	input_specs;
	// std::map<const std::string,data_spec_ptr> 	label_specs;
	// std::map<const std::string,data_spec_ptr> 	mask_specs;
	std::list<data_spec_ptr> 	input_specs;
	std::list<data_spec_ptr> 	label_specs;
	std::list<data_spec_ptr> 	mask_specs;


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
			data_spec_ptr spec = data_spec_ptr(new data_spec(name));
			STRONG_ASSERT( spec->build(config_) );

			if ( std::string::npos != name.find("INPUT") )
			{				
				// input_specs[name] = spec;
				input_specs.push_back(spec);
			}
			else if ( std::string::npos != name.find("LABEL") )
			{
				// label_specs[name] = spec;
				label_specs.push_back(spec);
			}
			else if ( std::string::npos != name.find("MASK") )
			{
				// mask_specs[name] = spec;
				mask_specs.push_back(spec);
			}
		}

		return true;
	}


public:
	data_spec_parser( const std::string& config )
		: config_(config)
	{
		if ( !parse_config() )
		{
			std::string what 
				= "Failed to parse data spec file [" + config + "]";
			throw std::invalid_argument(what);
		}
	}

}; // class data_spec_parser

}} // namespace zi::znn

#endif // ZNN_DATA_SPEC_PARSER_HPP_INCLUDED