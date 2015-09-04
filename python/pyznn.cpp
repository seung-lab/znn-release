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

// znn
#include "network/parallel/network.hpp"
#include "cube/cube.hpp"
#include <zi/zargs/zargs.hpp>

//utils
#include "pyznn_utils.hpp"

namespace bp = boost::python;
namespace np = boost::numpy;
using namespace znn::v4;
using namespace znn::v4::parallel_network;

std::shared_ptr< network > CNet_Init(
		std::string  const net_config_file,
		np::ndarray  const & outsz_a,
		std::size_t  const tc,
        std::uint8_t const phs = 0) // 0:TRAIN, 1:TEST
{
    std::vector<options> nodes;
    std::vector<options> edges;
    parse_net_file(nodes, edges, net_config_file);
    vec3i out_sz(   reinterpret_cast<std::int64_t*>(outsz_a.get_data())[0],
                    reinterpret_cast<std::int64_t*>(outsz_a.get_data())[1],
                    reinterpret_cast<std::int64_t*>(outsz_a.get_data())[2]
					);

    // construct the network class
    std::shared_ptr<network> net(
        new network(nodes,edges,out_sz,tc,static_cast<phase>(phs)));

    net->optimize(nodes, edges, out_sz, tc, 10);

    return net;
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

//IO FUNCTIONS

//Returns a tuple of list of dictionaries of the following form
// (node_opts, edge_opts)
// node_opts = [node_group_option_dict, ...]
// edge_opts = [edge_group_option_dict, ...]
// See pyznn_utils.hpp for helper functions
bp::tuple CNet_getopts( bp::object const & self )
{
	network& net = bp::extract<network&>(self)();
	//Grabbing "serialized" options
	//opts.first => node options
	//opts.second => edge options
	std::pair<std::vector<options>,std::vector<options>> opts = net.serialize();

	//Init
	bp::list node_opts;
	bp::list edge_opts;

	//Node options
	for ( std::size_t i=0; i < opts.first.size(); i++ )
	{
		//Convert the map to a python dict, and append it
		node_opts.append( node_opt_to_dict(opts.first[i], self) );
	}

	//Derive size layer dictionary from node opts
	std::map<std::string, std::size_t> layer_sizes = extract_layer_sizes( opts.first );

	//Edge opts
	for ( std::size_t i=0; i < opts.second.size(); i++ )
	{
		//Convert the map to a python dict, and append it
		edge_opts.append( edge_opt_to_dict(opts.second[i], layer_sizes, self) );
	}

	return bp::make_tuple(node_opts, edge_opts);
}

//Initializes a CNet instance based on the passed options
std::shared_ptr<network> CNet_loadopts( bp::tuple const & opts,
	std::string const net_config_file,
	np::ndarray const & outsz_a,
	std::size_t const tc )
{

	bp::list node_opts_list = bp::extract<bp::list>( opts[0] );
	bp::list edge_opts_list = bp::extract<bp::list>( opts[1] );

	//See pyznn_utils.hpp
	std::vector<options> node_opts = pyopt_to_znnopt(node_opts_list);
	std::vector<options> edge_opts = pyopt_to_znnopt(edge_opts_list);

	vec3i out_sz(	reinterpret_cast<std::int64_t*>(outsz_a.get_data())[0],
					reinterpret_cast<std::int64_t*>(outsz_a.get_data())[1],
					reinterpret_cast<std::int64_t*>(outsz_a.get_data())[2]
					);	

	std::shared_ptr<network> net(
		new network(node_opts,edge_opts,out_sz,tc));
	
	return net;
}

BOOST_PYTHON_MODULE(pyznn)
{
    Py_Initialize();
    np::initialize();

    bp::class_<network, std::shared_ptr<network>, boost::noncopyable>("CNet",bp::no_init)
        .def("__init__", bp::make_constructor(&CNet_Init))
        .def("__init__", bp::make_constructor(&CNet_loadopts))
        .def("get_fov",     		&CNet_fov)
        .def("forward",     		&CNet_forward)
        .def("backward",			&CNet_backward)
        .def("set_eta",    			&network::set_eta)
        .def("set_phase",              &network::set_phase)
        .def("set_momentum",		&network::set_momentum)
        .def("set_weight_decay",	&network::set_weight_decay )
        .def("get_input_num", 		&CNet_get_input_num)
        .def("get_output_num", 		&CNet_get_output_num)
		.def("get_opts",			&CNet_getopts)
        ;
}
