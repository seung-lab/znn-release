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
        ;
}
