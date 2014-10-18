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

#ifndef ZNN_NODE_SPEC_HPP_INCLUDED
#define ZNN_NODE_SPEC_HPP_INCLUDED

#include <boost/program_options.hpp>


namespace zi {
namespace znn {

class node_spec
{
public:
	const std::string	name;
	std::size_t			size;			// number of nodes

	std::string			activation;		// activation function type
	std::string			act_params;		// parameters for activation function

	std::string 		filter;			// filtering type (e.g., max, average, etc.)
	vec3i 				filter_size;	// size of filtering kernel
	vec3i				filter_stride;	// stride of filtering kernel	

	std::string 		init_type;
	std::string 		init_params;

	double				bias;			// initial bias
	double 				eta;			// learning rate
	double 				mom;			// momentum
	double 				wc;				// weight decay

	bool				fft;			// receives fft
	bool				load;

private:
	// for parsing configuration file
	boost::program_options::options_description desc_;
	

public:
	#define TEXT_WRITE 	(std::ios::out)
	#define TEXT_READ	(std::ios::in)

	void save( const std::string& path ) const
	{
		std::string fname = path + name + ".spec";
        std::ofstream fout(fname.c_str(), TEXT_WRITE);

        fout << "[" << name << "]" << std::endl;
        fout << "size=" << size << std::endl;
        fout << "activation=" << activation << std::endl;
        fout << "act_params=" << act_params << std::endl;
        fout << "filter=" << filter << std::endl;
        fout << "filter_size=" << vec3i_to_string(filter_size) << std::endl;
        fout << "filter_stride=" << vec3i_to_string(filter_stride) << std::endl;
        fout << "init_type=" << init_type << std::endl;
        fout << "init_params=" << init_params << std::endl;
        fout << "bias=" << bias << std::endl;
        fout << "eta=" << eta << std::endl;
        fout << "mom=" << mom << std::endl;
        fout << "wc=" << wc << std::endl;
        fout << "fft=" << fft << std::endl;
        // Don't save load
        // fout << "load=" << load << std::endl;

        fout.close();
	}

	bool build( const std::string& fpath )
	{
        std::ifstream fin(fpath.c_str(), TEXT_READ);
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
	vec3i real_filter_size( const vec3i& sparse ) const
    {
    	return (filter_size - vec3i::one)*sparse + vec3i::one;
    }


private:
	void initialize()
	{
		using namespace boost::program_options;

		desc_.add_options()
			((name+".size").c_str(),value<std::size_t>(&size)->default_value(1),"size")
	        ((name+".activation").c_str(),value<std::string>(&activation)->default_value("logistic"),"activation")
	        ((name+".act_params").c_str(),value<std::string>(&act_params)->default_value(""),"act_params")
	        ((name+".filter").c_str(),value<std::string>(&filter)->default_value("max"),"filter")
	        ((name+".filter_size").c_str(),value<std::string>()->default_value("1,1,1"),"filter_size")
	        ((name+".filter_stride").c_str(),value<std::string>()->default_value("1,1,1"),"filter_stride")
	        ((name+".init_type").c_str(),value<std::string>(&init_type)->default_value("zero"),"init_type")
	        ((name+".init_params").c_str(),value<std::string>(&init_params)->default_value(""),"init_params")
	        ((name+".bias").c_str(),value<double>(&bias)->default_value(0.0),"bias")
	        ((name+".eta").c_str(),value<double>(&eta)->default_value(0.01),"eta")
	        ((name+".mom").c_str(),value<double>(&mom)->default_value(0.0),"mom")
	        ((name+".wc").c_str(),value<double>(&wc)->default_value(0.0),"wc")	        
	        ((name+".fft").c_str(),value<bool>(&fft)->default_value(false),"fft")
	        ((name+".load").c_str(),value<bool>(&load)->default_value(true),"load")
	        ;
	}

	void postprocess( boost::program_options::variables_map& vm )
	{
		zi::zargs_::parser<std::vector<std::size_t> > _parser;
		std::vector<std::size_t> target;
		std::string source;

		// filter_size
		source = vm[name+".filter_size"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			filter_size = vec3i(target[0],target[1],target[2]);
		}

		// filter_stride
		target.clear();
		source = vm[name+".filter_stride"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			filter_stride = vec3i(target[0],target[1],target[2]);
		}
	}


public:
	node_spec( const std::string& _name )
		: name(_name)
	{
		initialize();
	}

}; // class node_spec

typedef boost::shared_ptr<node_spec> node_spec_ptr;

}} // namespace zi::znn

#endif // ZNN_NODE_SPEC_HPP_INCLUDED
