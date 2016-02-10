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

/*
Boost Python Interface for ZNNv4

 This program acts as a translation between the C++ back-end
 and the Python front-end. This is primarily achieved by defining a
 python 'CNet' object with a number of key attributes and functions.

Functions:

 CNet.forward() - computes the forward pass given a numpy array,
 	and returns a numpy array of the output

 CNet.backward() - computes the backward pass given a numpy array
 	of gradient values, implicitly updates the parameters of the
 	network

 CNet.get_fov() - returns a tuple which describes the field-of-view
 	of the network

 CNet.set_eta() - sets the learning rate

 CNet.set_phase() - currently have no clue what this does

 CNet.momentum() - sets the momentum constant - what proportion of the
 	past update defines the next

 CNet.set_weight_decay() - sets the weight decay

 CNet.get_input_num() - returns the number of 3d input volumes to the network

 CNet.get_output_num() - returns the number of 3d output volumes to the network

 CNet.get_opts() - serializes all fields of the network data structure, and returns
 	them as a tuple of lists of dictionaries. Each dictionary represents the fields
 	of a given layer of the network, the list consolidates all of the layers, and the
 	tuple separates node fields from edge fields.

 	This function is also used for saving networks to disk.

Jingpeng Wu <jingpeng.wu@gmail.com>
Nicholas Turner <nturner@cs.princeton.edu>, 2015
*/

//===========================================================================
// INCLUDE STATEMENTS
// boost python
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/lexical_cast.hpp>

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
#include "rand_error.hpp"

namespace bp = boost::python;
namespace np = boost::numpy;
using namespace znn::v4;
using namespace znn::v4::parallel_network;

//===========================================================================
//IO FUNCTIONS
//First constructor - generates a random network
std::shared_ptr< network > CNet_Init(
    std::string  const net_config_file,
    np::ndarray  const & outsz_a,
    std::size_t  tc  = 0,	// thread number
    bool const is_optimize = true,
    std::uint8_t const phs = 0, // 0:TRAIN, 1:TEST
    bool const force_fft = false)
{
    std::vector<options> nodes;
    std::vector<options> edges;
    std::cout<< "parse_net file: "<<net_config_file<<std::endl;
    parse_net_file(nodes, edges, net_config_file);
    vec3i out_sz(   reinterpret_cast<std::int64_t*>(outsz_a.get_data())[0],
                    reinterpret_cast<std::int64_t*>(outsz_a.get_data())[1],
                    reinterpret_cast<std::int64_t*>(outsz_a.get_data())[2]
        );
    if ( tc == 0 )
    	tc = std::thread::hardware_concurrency();

    // force fft or optimize
    if ( force_fft )
    {
        network::force_fft(edges);
    }
    else
    {
        if ( is_optimize )
        {
            phase _phs = static_cast<phase>(phs);
            if ( _phs == phase::TRAIN )
            {
                network::optimize(nodes, edges, out_sz, tc, 10);
            }
            else if ( _phs == phase::TEST )
            {
                network::optimize_forward(nodes, edges, out_sz, tc, 2);
            }
            else
            {
                std::string str = boost::lexical_cast<std::string>(phs);
                throw std::logic_error(HERE() + "unknown phase: " + str);
            }
        }
    }

    std::cout<< "construct the network class using the edges and nodes..." <<std::endl;
    // construct the network class
    std::shared_ptr<network> net(
        new network(nodes,edges,out_sz,tc,static_cast<phase>(phs)));

    return net;
}

//IO FUNCTIONS

//Second Constructor
//Initializes a CNet instance based on
// the passed options struct (tuple(list(dict)), see CNet_getopts)
std::shared_ptr<network> CNet_loadopts( bp::tuple const & opts,
                                        std::string const net_config_file,
                                        np::ndarray const & outsz_a,
                                        std::size_t const tc,
                                        bool const is_optimize = true,
                                        std::uint8_t const phs = 0,
                                        bool const force_fft = false )
{

    bp::list node_opts_list = bp::extract<bp::list>( opts[0] );
    bp::list edge_opts_list = bp::extract<bp::list>( opts[1] );

    //See pyznn_utils.hpp
    std::vector<options> node_opts = pyopt_to_znnopt(node_opts_list);
    std::vector<options> edge_opts = pyopt_to_znnopt(edge_opts_list);

    vec3i out_sz( reinterpret_cast<std::int64_t*>(outsz_a.get_data())[0],
                  reinterpret_cast<std::int64_t*>(outsz_a.get_data())[1],
                  reinterpret_cast<std::int64_t*>(outsz_a.get_data())[2]
            );

     // force fft or optimize
    if ( force_fft )
    {
        network::force_fft(edge_opts);
    }
    else
    {
        if ( is_optimize )
        {
            phase _phs = static_cast<phase>(phs);
            if ( _phs == phase::TRAIN )
            {
                network::optimize(node_opts, edge_opts, out_sz, tc, 10);
            }
            else if ( _phs == phase::TEST )
            {
                network::optimize_forward(node_opts, edge_opts, out_sz, tc, 2);
            }
            else
            {
                std::string str = boost::lexical_cast<std::string>(phs);
                throw std::logic_error(HERE() + "unknown phase: " + str);
            }
        }
    }

    std::shared_ptr<network> net(
        new network( node_opts,edge_opts,out_sz,tc,static_cast<phase>(phs) ));

    return net;
}

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

//===========================================================================
//PROPOGATION FUNCTIONS

//Computes the forward-pass
bp::dict CNet_forward( bp::object const & self, bp::dict& ins )
{
    // extract the class from self
    network& net = bp::extract<network&>(self)();

    // run forward and get output
    auto prop = net.forward( std::move( pydict2sample<real>( ins ) ) );
    return sample2pydict<real>( self, prop);
}

//Computes the backward-pass and updates network parameters
void CNet_backward( bp::object & self, bp::dict& grdts )
{
    // extract the class from self
    network& net = bp::extract<network&>(self)();

    // setting up output sample
    std::map<std::string, std::vector<cube_p<real>>> gsample;
    gsample = pydict2sample<real>( grdts );

    // backward
    net.backward( std::move(gsample) );
}

//===========================================================================
//NETWORK STATISTIC FUNCTIONS

//Returns the field-of-view as a tuple
bp::tuple CNet_fov( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    vec3i fov_vec =  net.fov();
    return bp::make_tuple(fov_vec[0], fov_vec[1], fov_vec[2]);
}

//Returns the number of 3d input volumes for the network
std::size_t CNet_get_input_num( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    std::map<std::string, std::pair<vec3i, std::size_t>> ins = net.inputs();
    return ins["input"].second;
}

//Returns the number of 3d output volumes for the network
std::size_t CNet_get_output_num( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    std::map<std::string, std::pair<vec3i,std::size_t>> outs = net.outputs();
    return outs["output"].second;
}

bp::dict CNet_get_inputs_setsz( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    std::map<std::string, std::pair<vec3i,size_t>> inputs = net.inputs();

    bp::dict ret;
    for (auto &in: inputs)
    {
        np::ndarray arr = np::empty(bp::make_tuple(4), np::dtype::get_builtin<unsigned int>());
        arr[0] = in.second.second;
        arr[1] = in.second.first[0];
        arr[2] = in.second.first[1];
        arr[3] = in.second.first[2];
        ret[in.first] = arr;
    }
    return ret;
}

bp::dict CNet_get_outputs_setsz( bp::object const & self )
{
    network& net = bp::extract<network&>(self)();
    std::map<std::string, std::pair<vec3i,size_t>> outputs = net.outputs();

    bp::dict ret;
    for (auto &out: outputs)
    {
        np::ndarray arr = np::empty(bp::make_tuple(4), np::dtype::get_builtin<unsigned int>());
        arr[0] = out.second.second;
        arr[1] = out.second.first(0);
        arr[2] = out.second.first(1);
        arr[3] = out.second.first(2);
        ret[out.first] = arr;
    }
    return ret;
}
//===========================================================================
void CNet_set_phase(bp::object const & self, std::uint8_t const phs = 0)
{
    network& net = bp::extract<network&>(self)();
    net.set_phase( static_cast<phase>(phs) );
    return;
}

//===========================================================================
real pyget_rand_error(np::ndarray& affs_arr, np::ndarray& taffs_arr)
{
    std::vector<cube_p<real>> taffs = array2cubelist<real>( taffs_arr );
    std::vector<cube_p<real>> affs  = array2cubelist<real>( affs_arr );

    real re = get_rand_error( affs, taffs);
    return re;
}

//===========================================================================
//BOOST PYTHON INTERFACE DEFINITION
BOOST_PYTHON_MODULE(pyznn)
{
    Py_Initialize();
    np::initialize();

    bp::class_<network, boost::shared_ptr<network>, boost::noncopyable>("CNet",bp::no_init)
        .def("__init__", bp::make_constructor(&CNet_Init))
        .def("__init__", bp::make_constructor(&CNet_loadopts))
        .def("get_fov",  &CNet_fov)
        .def("forward",  &CNet_forward)
        .def("backward", &CNet_backward)
        .def("set_eta",                 &network::set_eta)
        .def("set_phase",               &CNet_set_phase)
        .def("set_momentum",		&network::set_momentum)
        .def("set_weight_decay",	&network::set_weight_decay )
        .def("get_inputs_setsz", 	&CNet_get_inputs_setsz)
        .def("get_input_num", 		&CNet_get_input_num)
        .def("get_outputs_setsz", 	&CNet_get_outputs_setsz)
        .def("get_output_num", 		&CNet_get_output_num)
        .def("get_opts",		&CNet_getopts)
        ;
    def("get_rand_error", pyget_rand_error);
}
