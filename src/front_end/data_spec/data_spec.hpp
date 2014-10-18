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

#ifndef ZNN_DATA_SPEC_HPP_INCLUDED
#define ZNN_DATA_SPEC_HPP_INCLUDED

#include <zi/zargs/parser.hpp>

#include <boost/program_options.hpp>


namespace zi {
namespace znn {

class data_spec
{
public:
	const std::string	name;
	std::string 		path;		// file path
	std::string 		ext;		// file extension (path.ext)
	vec3i 				size;		// data size
	vec3i				offset;		// offset w.r.t. global origin

	std::string			pptype;		// preprocess type
	std::string			ppargs;		// preprocess parameters

private:
	// for parsing configuration file
	boost::program_options::options_description desc_;
	

public:	
	#define TEXT_WRITE 	(std::ios::out)
	#define TEXT_READ	(std::ios::in)

	bool build( const std::string& fpath )
	{
		std::ifstream fin(fpath.c_str(),TEXT_READ);
        if ( !fin )	return false;

        namespace po = boost::program_options;
        po::variables_map vm;
        po::store(po::parse_config_file(fin,desc_,true),vm);
        po::notify(vm);
        postprocess(vm);

        fin.close();
        return true;
	}


public:
	std::string get_path() const
	{
		std::string ret = path;
		if ( !ext.empty() )
		{
			ret = ret + "." + ext;
		}
		return ret;
	}


private:
	void initialize()
	{
		using namespace boost::program_options;

		desc_.add_options()			
	        ((name+".path").c_str(),value<std::string>(&path)->default_value(""),"path")
	        ((name+".ext").c_str(),value<std::string>(&ext)->default_value(""),"ext")
	        ((name+".size").c_str(),value<std::string>()->default_value("0,0,0"),"size")
	        ((name+".offset").c_str(),value<std::string>()->default_value("0,0,0"),"offset")
	        ((name+".pptype").c_str(),value<std::string>(&pptype)->default_value(""),"pptype")
	        ((name+".ppargs").c_str(),value<std::string>(&ppargs)->default_value(""),"ppargs")
	        ;
	}

	void postprocess( boost::program_options::variables_map& vm )
	{
		zi::zargs_::parser<std::vector<std::size_t> > _parser;
		std::vector<std::size_t> target;
		std::string source;

		// path check
		if ( !get_path().empty() )
		{
			boost::filesystem::path data_path(get_path());

			if ( !boost::filesystem::exists(data_path) )
			{
				std::string what = "Non-existent data path [" + get_path() + "]";
				throw std::invalid_argument(what);
			}

			if ( boost::filesystem::is_directory(data_path) )
			{
				std::string what = "Non-file data path [" + get_path() + "]";
				throw std::invalid_argument(what);
			}
		}

		// size
		source = vm[name+".size"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			for ( std::size_t i = 0; i < target.size(); ++i )
			{
				size[i] = target[i];
			}
		}

		// read size info. from file (.size)
		if ( size == vec3i::zero )
		{
			size = import_size_info(path);
		}

		if ( size[0]*size[1]*size[2] == 0 )
		{
			std::string what = "Bad data size [" + vec3i_to_string(size) + "]";
			throw std::invalid_argument(what);
		}

		// offset
		target.clear();
		source = vm[name+".offset"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			for ( std::size_t i = 0; i < target.size(); ++i )
			{
				offset[i] = target[i];
			}
		}
	}


public:
	friend std::ostream&
	operator<<( std::ostream& os, const data_spec& rhs )
	{
		return (os << "[" << rhs.name << "]" << '\n'
        		   << "path=" << rhs.path << '\n'
        		   << "ext=" << rhs.ext << '\n'
        		   << "size=" << vec3i_to_string(rhs.size) << '\n'
        		   << "offset=" << vec3i_to_string(rhs.offset) << '\n'
        		   << "pptype=" << rhs.pptype << '\n'
        		   << "ppargs=" << rhs.ppargs << '\n');
	}


public:
	data_spec( const std::string& _name )
		: name(_name)
		, size(vec3i::zero)
		, offset(vec3i::zero)
	{
		initialize();
	}

}; // class data_spec

typedef boost::shared_ptr<data_spec> data_spec_ptr;

}} // namespace zi::znn

#endif // ZNN_DATA_SPEC_HPP_INCLUDED
