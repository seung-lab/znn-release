#ifndef ZNN_EDGE_SPEC_HPP_INCLUDED
#define ZNN_EDGE_SPEC_HPP_INCLUDED

#include <zi/zargs/parser.hpp>

#include <boost/program_options.hpp>

namespace zi {
namespace znn {

class edge_spec
{
public:
	const std::string	name;
	vec3i				size;			// filter size

	std::string			init_type;		// weight initialization type
	std::string			init_params;	// parameters for weight initialization	

	double 				eta;			// learning rate
	double 				mom;			// momentum
	double 				wc;				// weight decay

	bool				load;

private:
	// for parsing configuration file
	boost::program_options::options_description desc_;
	

public:
	#define TEXT_WRITE 	(std::ios::out)
	#define TEXT_READ	(std::ios::in)

	void save( const std::string& path ) const
	{
		std::string fpath = path + name + ".spec";
        std::ofstream fout(fpath.c_str(), TEXT_WRITE);

        fout << "[" << name << "]" << std::endl;
        fout << "size=" << vec3i_to_string(size) << std::endl;
        fout << "init_type=" << init_type << std::endl;        
        fout << "init_params=" << init_params << std::endl;
        fout << "eta=" << eta << std::endl;
        fout << "mom=" << mom << std::endl;
        fout << "wc=" << wc << std::endl;

        fout.close();
	}

	bool build( const std::string& path )
	{		
        std::ifstream fin(path.c_str(), TEXT_READ);
        if ( !fin ) return false;
        
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
    	return (size - vec3i::one)*sparse + vec3i::one;
    }


private:
	void initialize()
	{
		using namespace boost::program_options;

		desc_.add_options()
			((name+".size").c_str(),value<std::string>()->default_value("1,1,1"),"size")
	        ((name+".init_type").c_str(),value<std::string>(&init_type)->default_value("zero"),"init_type")
	        ((name+".init_params").c_str(),value<std::string>(&init_params)->default_value(""),"init_params")
	        ((name+".eta").c_str(),value<double>(&eta)->default_value(0.0),"eta")
	        ((name+".mom").c_str(),value<double>(&mom)->default_value(0.0),"mom")
	        ((name+".wc").c_str(),value<double>(&wc)->default_value(0.0),"wc")
	        ((name+".load").c_str(),value<bool>(&load)->default_value(true),"load")
	        ;
	}

	void postprocess( boost::program_options::variables_map& vm )
	{
		zi::zargs_::parser<std::vector<std::size_t> > _parser;
		std::vector<std::size_t> target;
		std::string source;
		
		// size
		source = vm[name+".size"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			size = vec3i(target[0],target[1],target[2]);
		}
	}


public:
	edge_spec( const std::string& _name )
		: name(_name)
	{
		initialize();
	}

}; // class edge_spec

typedef boost::shared_ptr<edge_spec> edge_spec_ptr;

}} // namespace zi::znn

#endif // ZNN_EDGE_SPEC_HPP_INCLUDED
