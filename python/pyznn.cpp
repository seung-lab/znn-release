//
//// Copyright (C) 2015  Jinppeng Wu <jingpeng@princeton.edu>
//// ----------------------------------------------------------
////
//// This program is free software: you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation, either version 3 of the License, or
//// (at your option) any later version.
////
//// This program is distributed in the hope that it will be useful,
//// but WITHOUT ANY WARRANTY; without even the implied warranty of
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//// GNU General Public License for more details.
////
//// You should have received a copy of the GNU General Public License
//// along with this program.  If not, see <http://www.gnu.org/licenses/>.
////
// ----------------------------NOTE-----------------------------//
// both znn and python use C-order                              //
// znn v4 use x,y,z and z is changing the fastest               //
// python code use z,y,x and x is changing the fastest          //
// we just match znn(x,y,z) and python(z,y,x) directly,         //
// so the z in python matches the x in znn!!!                   //
// -------------------------------------------------------------//

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
//#include <H5Cpp.h>
// znn
#include "network/parallel/network.hpp"
#include "cube/cube.hpp"
#include <zi/zargs/zargs.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;
using namespace znn::v4;
using namespace znn::v4::parallel_network;


std::shared_ptr< network > CNet_Init(
    std::string net_config_file, std::int64_t outz,
    std::int64_t outy, std::int64_t outx, std::size_t tc )
{
    std::vector<options> nodes;
    std::vector<options> edges;
    parse_net_file(nodes, edges, net_config_file);
    vec3i out_sz(outz, outy, outx);

    // construct the network class
    std::shared_ptr<network> net(
        new network(nodes,edges,out_sz,tc));
    return net;
}

template <typename T>
cube_p<T> array2cube_p( np::ndarray array)
{
	// input volume size
	std::size_t sz = array.shape(0);
	std::size_t sy = array.shape(1);
	std::size_t sx = array.shape(2);

	// copy data to avoid the pointer free error
	cube_p<T> ret = get_cube<T>(vec3i(sz,sy,sx));
	for (std::size_t k=0; k<sz*sy*sx; k++)
		ret->data()[k] = reinterpret_cast<T*>(array.get_data())[k];
	return ret;
}

bp::list CNet_forward( bp::object const & self, bp::list& inarrays )
{
	// extract the class from self
	network& net = bp::extract<network&>(self)();

	// number of input arrays
	std::size_t len = boost::python::extract<std::size_t>(inarrays.attr("__len__")());

	// setting up input sample
	std::map<std::string, std::vector<cube_p< real >>> insample;
	insample["input"].resize(len);
	for (std::size_t i=0; i<len; i++)
	{
		np::ndarray array = bp::extract<np::ndarray>( inarrays[i] );
		insample["input"][i] = array2cube_p<real>( array );
	}

#ifndef NDEBUG
//	std::cout<<"lenth of input list of arrays: "<<len<<std::endl;
	// print the whole input cube
//    cube_p<real> in_p = insample["input"][0];
//	std::cout<<"input in c++: "<<std::endl;
//	for(std::size_t i=0; i< in_p->num_elements(); i++)
//		std::cout<<in_p->data()[i]<<", ";
//	std::cout<<std::endl;
#endif

    // run forward and get output
    auto prop = net.forward( std::move(insample) );

    // initalize the return list
    bp::list ret;

    // number of output cubes
    std::size_t num_out = prop["output"].size();

#ifndef NDEBUG
    std::cout<<"number of output cubes: "<<num_out<<std::endl;
#endif

    for (std::size_t i=0; i<num_out; i++)
    {
    	cube_p<real> out_cube_p = prop["output"][i];
    	// output size assert
		vec3i outsz( out_cube_p->shape()[0], out_cube_p->shape()[1], out_cube_p->shape()[2] );

#ifndef NDEBUG
		std::cout<<"output size: "<<outsz[0]<<"x"<<outsz[1]<<"x"<<outsz[2]<<std::endl;
#endif

    	// return ndarray
		np::ndarray outarray = 	np::from_data(
									out_cube_p->data(),
									np::dtype::get_builtin<real>(),
									bp::make_tuple(outsz[0],outsz[1],outsz[2]),
									bp::make_tuple(outsz[1]*outsz[2]*sizeof(real), outsz[2]*sizeof(real), sizeof(real)),
									self
								);
		ret.append( outarray );
    }

#ifndef NDEBUG

	// input volume size
    np::ndarray inarray = bp::extract<np::ndarray>( inarrays[0] );
	std::size_t sz = inarray.shape(0);
	std::size_t sy = inarray.shape(1);
	std::size_t sx = inarray.shape(2);
	vec3i insz( sz, sy, sx );
	vec3i fov = net.fov();
	vec3i outsz( prop["output"][0]->shape()[0], prop["output"][0]->shape()[1], prop["output"][0]->shape()[2] );
	assert(outsz == insz - fov + 1);
	std::cout<<"output size: "  <<outsz[0]<<"x"<<outsz[1]<<"x"<<outsz[2]<<std::endl;
	// print the whole output cube
	std::cout<<"output in c++: "<<std::endl;
	for(std::size_t i=0; i< prop["output"][0]->num_elements(); i++)
		std::cout<<prop["output"][0]->data()[i]<<", ";
	std::cout<<std::endl;
#endif

	return ret;
}

void CNet_backward( bp::object & self, bp::list& grads )
{
	// extract the class from self
	network& net = bp::extract<network&>(self)();

	// number of gradient volume
	std::size_t len = boost::python::extract<std::size_t>(grads.attr("__len__")());
	// setting up output sample
	std::map<std::string, std::vector<cube_p<real>>> outsample;
	outsample["output"].resize(len);
	for (std::size_t i=0; i<len; i++)
		outsample["output"][i] = array2cube_p<real>( bp::extract<np::ndarray>( grads[i] ));

#ifndef NDEBUG
	// print the whole gradient cube

//	cube_p<real> grdt_p = array2cube_p( grad );
//	std::cout<<"gradient from python: "<<std::endl;
//		for(std::size_t i=0; i< grdt_p->num_elements(); i++)
//			std::cout<<reinterpret_cast<real*>(grad.get_data())[i]<<", ";
//		std::cout<<std::endl;
//	std::cout<<"gradient in c++: "<<std::endl;
//	for(std::size_t i=0; i< grdt_p->num_elements(); i++)
//		std::cout<<grdt_p->data()[i]<<", ";
//	std::cout<<std::endl;
#endif

// backward
        net.backward( std::move(outsample) );
        return;
}

bp::tuple CNet_fov( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    vec3i fov_vec =  net.fov();
    return 	bp::make_tuple(fov_vec[0], fov_vec[1], fov_vec[2]);
}

std::size_t CNet_get_input_num( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    std::map<std::string, std::pair<vec3i, std::size_t>> ins = net.inputs();
    return ins["input"].second;
}

std::size_t CNet_get_output_num( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    std::map<std::string, std::pair<vec3i,std::size_t>> outs = net.outputs();
    return outs["output"].second;
}

//IO HELPER FUNCTIONS

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
}

//Takes a binary string, and converts it to a numpy array
bp::tuple bias_string_to_np( std::string const & bin, 
	std::vector<std::size_t> size,
	bp::object const & self )
{
	real const * data = reinterpret_cast<real const *>(bin.data());

	//momentum values stored immediately after array values
	std::size_t gap = bin.size() / (2 * sizeof(real));
	real const * momentum = data + gap;

	return bp::make_tuple(
		//values
		np::from_data(data,
					np::dtype::get_builtin<real>(),
					bp::make_tuple(size[0]),
					bp::make_tuple(sizeof(real)),
					self
					),
		//momentum values
		np::from_data(momentum,
					np::dtype::get_builtin<real>(),
					bp::make_tuple(size[0]),
					bp::make_tuple(sizeof(real)),
					self
					)
		);
}


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
					),
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
					)
		);
}
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
}

//znn::options -> dict
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
		if (p.first == "biases" || p.first == "filters" )
		{
			//Debug
			// res["raw_biases"] = p.second;
			res[p.first] = bias_string_to_np(p.second, size, self);
		}
	}
	return res;
}

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
}

//Edge version, also takes the layer_sizes dict necessary to import filters
// properly
bp::dict edge_opt_to_dict( options const opt, 
	std::map<std::string, std::size_t> layer_sizes,
	bp::object const & self )
{
	bp::dict res;
	std::vector<std::size_t> size;
	std::string input = "";
	std::string output = "";

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
			input = p.second;
			res[p.first] = p.second;
		}
		else if ( p.first == "output" )
		{
			output = p.second;
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
			// res["raw_biases"] = p.second;
			std::size_t nodes_in = layer_sizes[input];
			std::size_t nodes_out = layer_sizes[output];

			res[p.first] = filter_string_to_np(p.second, size, 
											nodes_in,
											nodes_out, 
											self);
		}
	}
	return res;
}


//IO FUNCTIONS

//Returns a list of 
bp::tuple CNet_getopts( bp::object const & self )
{
	network& net = bp::extract<network&>(self)();
	//Grabbing "serialized" options
	//opts.first => node options
	//opts.second => edge options
	std::pair<std::vector<options>,std::vector<options>> opts = net.serialize();

	//Debug
	// opts.first[1].dump();
	// std::cout<<std::endl;
	// opts.second[1].dump();
	// std::cout<<std::endl;

	//Init
	bp::list node_opts;
	bp::list edge_opts;

	//Node options
	for ( std::size_t i=0; i < opts.first.size(); i++ )
	{
		//Convert the map to a python dict, and append it
		node_opts.append( node_opt_to_dict(opts.first[i], self) );
	}

	//TO DO: Derive size layer dictionary from node opts
	std::map<std::string, std::size_t> layer_sizes = extract_layer_sizes( opts.first );

	//Edge opts
	for ( std::size_t i=0; i < opts.second.size(); i++ )
	{
		//Convert the map to a python dict, and append it
		edge_opts.append( edge_opt_to_dict(opts.second[i], layer_sizes, self) );
	}

	return bp::make_tuple(node_opts, edge_opts);
}

BOOST_PYTHON_MODULE(pyznn)
{
    Py_Initialize();
    np::initialize();

    bp::class_<network, std::shared_ptr<network>, boost::noncopyable>("CNet",bp::no_init)
        .def("__init__", bp::make_constructor(&CNet_Init))
        .def("get_fov",     		&CNet_fov)
		.def("forward",     		&CNet_forward)
		.def("backward",			&CNet_backward)
		.def("set_eta",    			&network::set_eta)
		.def("set_momentum",		&network::set_momentum)
		.def("set_weight_decay",	&network::set_weight_decay )
		.def("get_input_num", 		&CNet_get_input_num)
		.def("get_output_num", 		&CNet_get_output_num)
		.def("get_opts",				&CNet_getopts)
        ;
}
