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
		std::string const net_config_file,
		np::ndarray const & outsz_a,
		std::size_t const tc )
{
    std::vector<options> nodes;
    std::vector<options> edges;
    parse_net_file(nodes, edges, net_config_file);
    vec3i out_sz(	reinterpret_cast<std::int64_t*>(outsz_a.get_data())[0],
    				reinterpret_cast<std::int64_t*>(outsz_a.get_data())[1],
					reinterpret_cast<std::int64_t*>(outsz_a.get_data())[2]
					);
    // construct the network class
    std::shared_ptr<network> net(
        new network(nodes,edges,out_sz,tc));
    return net;
}

template <typename T>
std::vector<cube_p< T >> array2cubelist( np::ndarray& vols )
{
	// ensure that the input ndarray is 4 dimension
	assert( vols.get_nd() == 4 );

	std::vector<cube_p< T >> ret;
	ret.resize( vols.shape(0) );
	// input volume size
	std::size_t sz = vols.shape(1);
	std::size_t sy = vols.shape(2);
	std::size_t sx = vols.shape(3);

#ifndef NDEBUG
			std::cout<<"array2cubelist:"<<std::endl;
#endif

	for (std::size_t c=0; c<vols.shape(0); c++)
	{
		cube_p<T> cp = get_cube<T>(vec3i(sz,sy,sx));
		for (std::size_t k=0; k< sz*sy*sx; k++)
		{
			cp->data()[k] = reinterpret_cast<T*>( vols.get_data() )[c*sz*sy*sx + k];
#ifndef NDEBUG
			std::cout<<cp->data()[k]<<",";
#endif
		}
		ret[c] = cp;
	}
#ifndef NDEBUG
	std::cout<<std::endl<<"---------end--------"<<std::endl;
#endif
	return ret;
}

template <typename T>
np::ndarray cubelist2array( bp::object const & self, std::vector<cube_p< T >> clist )
{
	// number of output cubes
	std::size_t sc = clist.size();
	std::size_t sz = clist[0]->shape()[0];
	std::size_t sy = clist[1]->shape()[1];
	std::size_t sx = clist[2]->shape()[2];

	// temporal 4D qube pointer
	qube_p<T> tqp = get_qube<T>( vec4i(sc,sz,sy,sx) );
	for (std::size_t c=0; c<sc; c++)
	{
		for (std::size_t k=0; k<sz*sy*sx; k++)
			tqp->data()[c*sz*sy*sx+k] = clist[c]->data()[k];
	}
	// return ndarray
	return 	np::from_data(
				tqp->data(),
				np::dtype::get_builtin<T>(),
				bp::make_tuple(sc,sz,sy,sx),
				bp::make_tuple(sx*sy*sz*sizeof(T), sx*sy*sizeof(T), sx*sizeof(T), sizeof(T)),
				self
			);
}

np::ndarray CNet_forward( bp::object const & self, np::ndarray& inarrays )
{
	// extract the class from self
	network& net = bp::extract<network&>(self)();

	// setting up input sample
	std::map<std::string, std::vector<cube_p< real >>> insample;
	insample["input"] = array2cubelist<real>( inarrays );

    // run forward and get output
    auto prop = net.forward( std::move(insample) );

    return cubelist2array<real>(self, prop["output"]);
}

void CNet_backward( bp::object & self, np::ndarray& grdts )
{
	// extract the class from self
	network& net = bp::extract<network&>(self)();

	// setting up output sample
	std::map<std::string, std::vector<cube_p<real>>> outsample;
	outsample["output"] = array2cubelist<real>(grdts);

	// backward
    net.backward( std::move(outsample) );
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

<<<<<<< HEAD

=======
>>>>>>> d5c164099dbb26863a6fe717dde2094591b942ad
//Takes a comma delimited string (from option object), and converts it into
// a vector
std::vector<std::size_t> comma_delim_to_vector( std::string const comma_delim)
{
	//Debug
	std::cout << "tuple string" << comma_delim << std::endl;
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

<<<<<<< HEAD
bp::tuple string_to_np_array( std::string const & bin,
=======
bp::tuple string_to_np_array( std::string const & bin, 
>>>>>>> d5c164099dbb26863a6fe717dde2094591b942ad
	std::vector<std::size_t> size,
	bp::object const & self )
{
	real const * data = reinterpret_cast<real const *>(bin.data());

	if ( size.size() == 1 )
<<<<<<< HEAD
	{
=======
	{	
>>>>>>> d5c164099dbb26863a6fe717dde2094591b942ad
		return bp::make_tuple(
			//values
			np::from_data(data,
						np::dtype::get_builtin<real>(),
						bp::make_tuple(size[0]),
						bp::make_tuple(sizeof(real)),
						self
						),
			//momentum values
			np::from_data(data,
						np::dtype::get_builtin<real>(),
						bp::make_tuple(size[0]),
						bp::make_tuple(sizeof(real)),
						self
						)
			);

	}
	else //size.size() == 3
	{
		return bp::make_tuple(
			//values
			np::from_data(data,
						np::dtype::get_builtin<real>(),
						bp::make_tuple(size[0],size[1],size[2]),
						bp::make_tuple(size[1]*size[2]*sizeof(real), size[2]*sizeof(real), sizeof(real)),
						self
						),
			//momentum values
			np::from_data(data,
						np::dtype::get_builtin<real>(),
						bp::make_tuple(size[0],size[1],size[2]),
						bp::make_tuple(size[1]*size[2]*sizeof(real), size[2]*sizeof(real), sizeof(real)),
						self
						)
			);
	}
}

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
bp::dict opt_to_dict( options const opt, bp::object const & self )
{
	bp::dict res;
	std::vector<std::size_t> size;

<<<<<<< HEAD
	// First do a conversion of all fields except
=======
	//First do a conversion of all fields except
>>>>>>> d5c164099dbb26863a6fe717dde2094591b942ad
	// biases and filters
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
		else if ( p.first != "biases" && p.first != "filters" )
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
			res["raw_biases"] = p.second;
			res[p.first] = string_to_np_array(p.second, size, self);
		}
	}
	return res;
}

//IO
bp::tuple CNet_getopts( bp::object const & self )
{
	network& net = bp::extract<network&>(self)();
	//Grabbing "serialized" options
	//opts.first => node options
	//opts.second => edge options
	std::pair<std::vector<options>,std::vector<options>> opts = net.serialize();

	//Debug
	opts.first[1].dump();
	std::cout<<std::endl;
	opts.second[1].dump();
	std::cout<<std::endl;

	//Init
	bp::list node_opts;
	bp::list edge_opts;

	//Node options
	for ( std::size_t i=0; i < opts.first.size(); i++ )
	{
		//Convert the map to a python dict, and append it
		node_opts.append( opt_to_dict(opts.first[i], self) );
	}

	for ( std::size_t i=0; i < opts.second.size(); i++ )
	{
		//Convert the map to a python dict, and append it
		edge_opts.append( opt_to_dict(opts.second[i], self) );
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
<<<<<<< HEAD
        .def("forward",     		&CNet_forward)
        .def("backward",			&CNet_backward)
        .def("set_eta",    			&network::set_eta)
        .def("set_momentum",		&network::set_momentum)
        .def("set_weight_decay",	&network::set_weight_decay )
        .def("get_input_num", 		&CNet_get_input_num)
        .def("get_output_num", 		&CNet_get_output_num)
		.def("get_opts",			&CNet_getopts)
=======
		.def("forward",     		&CNet_forward)
		.def("backward",			&CNet_backward)
		.def("set_eta",    			&network::set_eta)
		.def("set_momentum",		&network::set_momentum)
		.def("set_weight_decay",	&network::set_weight_decay )
		.def("get_input_num", 		&CNet_get_input_num)
		.def("get_output_num", 		&CNet_get_output_num)
		.def("get_opts",				&CNet_getopts)
>>>>>>> d5c164099dbb26863a6fe717dde2094591b942ad
        ;
}
