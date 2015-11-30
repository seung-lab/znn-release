//===========================================================================
// INCLUDE STATEMENTS
// boost python
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/python/stl_iterator.hpp>

// system
#include <string>
#include <memory>
#include <cstdint>
#include <assert.h>
// znn
#include "network/parallel/network.hpp"

namespace bp = boost::python;
namespace np = boost::numpy;

namespace znn { namespace v4 {
//===========================================================================

//Converts std::vector<std::size_t> vector to tuple of std::size_t
// used to format size and stride options before passing to python
bp::tuple vec_to_tuple( std::vector<std::size_t> vec )
{
	if ( vec.size() == 1 )
	{
		return bp::make_tuple(vec[0]);
	}
	else //vec.size == 3
	{
		return bp::make_tuple(vec[0], vec[1], vec[2]);
	}
};

//Takes a comma delimited string (from option object), and converts it into
// a vector
// Used to convert strings like "1,7,7" to a numeric represenstation
std::vector<std::size_t> comma_delim_to_vector( std::string const comma_delim)
{
	//Debug
	// std::cout << "tuple string" << comma_delim << std::endl;
	//size can either be length 1 or length 3
	std::vector<std::size_t> res;

	std::string substring = comma_delim;

	while ( true )
	{
		std::size_t comma_ind = substring.find(',');

		if ( substring.find(',') == std::string::npos )
		{
			res.push_back( stoi(substring) );
			break;
		}

		res.push_back( stoi(substring.substr(0, comma_ind)) );
		substring = substring.substr(comma_ind+1);

	}
	return res;
};

void print_data_string( real const * data, std::size_t num_items )
{
	for ( std::size_t i=0; i < num_items; i++)
	{
		std::cout << data[i] << std::endl;
	}
	return;
};

//Takes a binary string, and converts it to a tuple of numpy arrays
// (bias values, momentum values)
//Assumes the intended array is one-dimensional (fitting for biases)
// and that momentum values are stored following the original values
bp::tuple bias_string_to_np( std::string const & bin,
	std::vector<std::size_t> size,
	bp::object const & self )
{
	real const * data = reinterpret_cast<real const *>(bin.data());

	//momentum values stored immediately after array values
	std::size_t gap = bin.size() / (2 * sizeof(real));
	real const * momentum = data + gap;

	//Debug
	//print_data_string(data, bin.size() / (2 * sizeof(real)));

	return bp::make_tuple(
		//values
		np::from_data(data,
					np::dtype::get_builtin<real>(),
					bp::make_tuple(size[0]),
					bp::make_tuple(sizeof(real)),
					self
					).copy(),
		//momentum values
		np::from_data(momentum,
					np::dtype::get_builtin<real>(),
					bp::make_tuple(size[0]),
					bp::make_tuple(sizeof(real)),
					self
					).copy()
		);
};

//Same thing for convolution filters
// Assumes the input size is THREE dimensional
bp::tuple filter_string_to_np( std::string const & bin,
	std::vector<std::size_t> size,
	std::size_t nodes_in,
	std::size_t nodes_out,
	bp::object const & self)
{
	real const * data = reinterpret_cast<real const *>(bin.data());

	//momentum values stored immediately after array values
	std::size_t gap = bin.size() / (2 * sizeof(real));
	real const * momentum = data + gap;

	//Debug
	//print_data_string(data, bin.size() / (2 * sizeof(real)));

	return bp::make_tuple(
		//values
		np::from_data(data,
					np::dtype::get_builtin<real>(),
					bp::make_tuple(nodes_in, nodes_out, size[0],size[1],size[2]),
					bp::make_tuple(nodes_out*size[0]*size[1]*size[2]*sizeof(real),
								   size[0]*size[1]*size[2]*sizeof(real),
								   size[1]*size[2]*sizeof(real),
								   size[2]*sizeof(real),
								   sizeof(real)),
					self
					).copy(),//copying seems to prevent overwriting
		//momentum values
		np::from_data(momentum,
					np::dtype::get_builtin<real>(),
					bp::make_tuple(nodes_in, nodes_out, size[0],size[1],size[2]),
					bp::make_tuple(nodes_out*size[0]*size[1]*size[2]*sizeof(real),
								   size[0]*size[1]*size[2]*sizeof(real),
								   size[1]*size[2]*sizeof(real),
								   size[2]*sizeof(real),
								   sizeof(real)),
					self
					).copy()//copying seems to prevent overwriting
		);
};

//Finds the number of nodes for all node groups specified within a vector
// of options. This is useful in importing the convolution filters
std::map<std::string, std::size_t> extract_layer_sizes( std::vector<options> opts )
{

	std::map<std::string, std::size_t> res;

	for ( std::size_t i=0; i < opts.size(); i++ )
	{
		std::string layer_name = opts[i]["name"];
		std::size_t layer_size = stoi(opts[i]["size"]);

		res[layer_name] = layer_size;
	}

	return res;
};

//znn::options -> dict for nodes (no need for input/output node sizes)
bp::dict node_opt_to_dict( options const opt,
	bp::object const & self )
{
	bp::dict res;
	std::vector<std::size_t> size;

	//First do a conversion of all fields except
	// biases and filters to gather necessary information
	// (size of filters, # input and output filters)
	for ( auto & p : opt )
	{
		if ( p.first == "size" )
		{
			size = comma_delim_to_vector(p.second);
			res[p.first] = vec_to_tuple(size);
		}
		else if ( p.first != "biases" )
		{
			res[p.first] = p.second;
		}
	}

	//Then scan again, for a field we can reshape into a np array
	for (auto & p : opt )
	{
		if ( p.first == "biases" )
		{
			//Debug
			// res["raw_biases"] = p.second;
			//std::cout << opt.require_as<std::string>("name") << std::endl;
			res[p.first] = bias_string_to_np(p.second, size, self);
		}
	}
	return res;
};

//Edge version, also takes the layer_sizes dict necessary to import filters
// properly
bp::dict edge_opt_to_dict( options const opt,
	std::map<std::string, std::size_t> layer_sizes,
	bp::object const & self )
{
	bp::dict res;
	std::vector<std::size_t> size;
	std::string input_layer = "";
	std::string output_layer = "";

	//First do a conversion of all fields except
	// biases and filters to gather necessary information
	// (size of filters, # input and output filters)
	for ( auto & p : opt )
	{
		if ( p.first == "size" )
		{
			size = comma_delim_to_vector(p.second);
			res[p.first] = vec_to_tuple(size);
		}
		else if ( p.first == "stride" )
		{
			res[p.first] = vec_to_tuple(comma_delim_to_vector(p.second));
		}
		else if ( p.first == "input" )
		{
			input_layer = p.second;
			res[p.first] = p.second;
		}
		else if ( p.first == "output" )
		{
			output_layer = p.second;
			res[p.first] = p.second;
		}
		else if ( p.first != "filters" )
		{
			res[p.first] = p.second;
		}
	}

	//Then scan again, for a field we can reshape into a np array
	for (auto & p : opt )
	{
		if (p.first == "filters" )
		{
			//Debug
			// res["raw_filters"] = p.second;
			//std::cout << opt.require_as<std::string>("name") << std::endl;
			std::size_t nodes_in = layer_sizes[input_layer];
			std::size_t nodes_out = layer_sizes[output_layer];

			res[p.first] = filter_string_to_np(p.second, size,
											nodes_in,
											nodes_out,
											self);
		}
	}
	return res;
};

//Loops over a bp::dict, and returns a vector of key strings
// used to define loop in pyopt_to_znnopt
std::vector<std::string> extract_dict_keys( bp::dict const & layer_dict )
{
	std::vector<std::string> res;
	std::size_t num_keys = bp::extract<std::size_t>( layer_dict.attr("__len__")() );

	for (std::size_t i=0; i < num_keys; i++ )
	{
		//fetching the keys through the bp interface...one at a time...
		res.push_back( bp::extract<std::string>(layer_dict.attr("keys")()[i]) );
	}

	return res;
}

//Creates a comma-delimited string out of a length-3 bp::tuple
// used to convert "size" or "stride" fields within opt_to_string
std::string comma_delim( bp::tuple const & tup )
{
	char buffer[20];
	int first = bp::extract<int>( tup[0] );
	int second = bp::extract<int>( tup[1] );
	int third = bp::extract<int>( tup[2] );

	snprintf( buffer, 20, "%d,%d,%d", first, second, third);

	return buffer;
}

//Takes a bp::dict field and converts the field to a string
// which we can store within a options struct (pyopt_to_znnopt)
std::string opt_to_string( bp::dict const & layer_dict, std::string key )
{
	if ( key == "size" )
	{ //either (#,) or (#,#,#)
		std::size_t tuple_len = bp::extract<std::size_t>(
			layer_dict[key].attr("__len__")() );

		if (tuple_len == 1)
		{
			std::size_t s = bp::extract<std::size_t>( layer_dict[key][0] );
			// std::cout << key << ": " << std::to_string(s) << std::endl;
			return std::to_string( s );
		} else  //tuple_len == 3
		{
			bp::tuple tup = bp::extract<bp::tuple>( layer_dict[key] );
			// std::cout << key << ": " << comma_delim(tup) << std::endl;
			return comma_delim( tup );
		}

	} else if ( key == "stride" )
	{ // (#,#,#)
		bp::tuple tup = bp::extract<bp::tuple>( layer_dict[key] );
		// std::cout << key << ": " << comma_delim(tup) << std::endl;
		return comma_delim( tup );

	} else if ( key == "biases" || key == "filters" )
	{ // (np.ndarray, np.ndarray)
		bp::tuple val_momentum = bp::extract<bp::tuple>( layer_dict[key] );
		std::string value_string = bp::extract<std::string>( val_momentum[0].attr("tostring")() );
		std::string momentum_string = bp::extract<std::string>( val_momentum[1].attr("tostring")() );

		// std::cout << key << ": " << value_string << momentum_string << std::endl;
		return value_string + momentum_string;
	} else
	{
		// std::string value = bp::extract<std::string>(layer_dict[key]);
		// std::cout << key << ": " << value << std::endl;
		return bp::extract<std::string>( layer_dict[key] );
	}
}

//Takes a list of dictionaries (pyopt)
// and converts it to a vector of znn options
std::vector<options> pyopt_to_znnopt( bp::list const & py_opts )
{

	std::vector<options> res;
	std::size_t num_layers = bp::extract<std::size_t>( py_opts.attr("__len__")() );

	//list loop
	for (std::size_t i=0; i < num_layers; i++ )
	{

		res.resize(res.size()+1);

		bp::dict layer_dict = bp::extract<bp::dict>( py_opts[i] );
		std::vector<std::string> keys = extract_dict_keys( layer_dict );

		//dict loop
		for (std::size_t k=0; k < keys.size(); k++ )
		{
			std::string opt_string = opt_to_string( layer_dict, keys[k] );
			res.back().push( keys[k], opt_string );
		}

	}

	return res;
}

template <typename T>
std::vector<cube_p< T >> array2cubelist( np::ndarray& vols )
{
    // ensure that the input ndarray is 4 dimension
    assert( vols.get_nd() == 4 );

    std::vector<cube_p< T >> ret;
    ret.resize( vols.shape(0) );
    // input volume size
    std::size_t sc = vols.shape(0);
    std::size_t sz = vols.shape(1);
    std::size_t sy = vols.shape(2);
    std::size_t sx = vols.shape(3);

    for (std::size_t c=0; c<sc; c++)
    {
        cube_p<T> cp = get_cube<T>(vec3i(sz,sy,sx));
        for (std::size_t k=0; k< sz*sy*sx; k++)
            cp->data()[k] = reinterpret_cast<T*>( vols.get_data() )[c*sz*sy*sx + k];
        ret[c] = cp;
    }
    return ret;
}

template <typename T>
std::map<std::string, std::vector<cube_p<T>>> pydict2sample( bp::dict pd)
{
    std::map<std::string, std::vector<cube_p<T>>> ret;
    bp::list keys = pd.keys();
    for (int i=0; i<bp::len(keys); i++ )
    {
        std::string key = bp::extract<std::string>( keys[i] );
        np::ndarray value = bp::extract<np::ndarray>( pd[key] );
        ret[key] = array2cubelist<T>( value );
    }
    return ret;
}

template <typename T>
inline np::ndarray
cubelist2array( bp::object const & self, std::vector<cube_p< T >> clist )
{
    // number of output cubes
    std::size_t sc = clist.size();
    std::size_t sz = clist[0]->shape()[0];
    std::size_t sy = clist[0]->shape()[1];
    std::size_t sx = clist[0]->shape()[2];

    // temporal 4D qube pointer
    qube_p<T> tqp = get_qube<T>( vec4i(sc,sz,sy,sx) );
    for (std::size_t c=0; c<sc; c++)
    {
        for (std::size_t k=0; k<sz*sy*sx; k++)
            tqp->data()[c*sz*sy*sx+k] = clist[c]->data()[k];
    }

    // return ndarray
    np::ndarray arr = np::from_data(
        tqp->data(),
        np::dtype::get_builtin<T>(),
        bp::make_tuple(sc,sz,sy,sx),
        bp::make_tuple(sx*sy*sz*sizeof(T), sx*sy*sizeof(T),
                       sx*sizeof(T), sizeof(T)),
        self );
    return arr.copy();
}


template <typename T>
bp::dict sample2pydict( bp::object const & self,
                        std::map<std::string, std::vector<cube_p<T>>> sample)
{
    bp::dict ret;
    for (auto & am: sample )
    {
        ret[am.first] = cubelist2array<T>( self, am.second);
    }
    return ret;
}


//NOT IMPORTANT YET (another version of masked training)
// may implement layer if I have time
// template <typename T>
// cube_p<T> x_chainvol( cube_p const & label )
// {
// 	vec3i shape = label.size();

// 	cube_p<T> chainvol = get_cube<T>(shape);

// 	for (std::size_t i=0; i < shape[0]; i++)
// 	{
// 		for (std::size_t j=0; i < shape[1]; j++)
// 		{
// 			for (std::size_t k=0; k < shape[2]; k++)
// 			{

// 			}
// 		}
// 	}
// }


}} //namespace znn::v4
