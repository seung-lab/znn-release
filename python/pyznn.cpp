
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
// ----------------------------NOTE---------------------------------//
// both znn and python use C-order									//
// znn v4 use x,y,z and z is changing the fastest					//
// python code use z,y,x and x is changing the fastest				//
// we just match znn(x,y,z) and python(z,y,x) directly,				//
// so the z in python matches the x in znn!!!						//
// -----------------------------------------------------------------//
// boost python
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

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

np::ndarray CNet_forward( bp::object const & self, const np::ndarray& inarray )
{
	// extract the class from self
	network& net = boost::python::extract<network&>(self)();

	// input volume size
	std::size_t sz = inarray.shape(0);
	std::size_t sy = inarray.shape(1);
	std::size_t sx = inarray.shape(2);
    
    // copy data to avoid the pointer free error
    cube_p<real> incube_p = get_cube<real>(vec3i(sz,sy,sx));
    for (std::size_t k=0; k<sz*sy*sx; k++)
        incube_p->data()[k] = inarray.get_data()[k];

    std::cout<<"put incube to insample..."<<std::endl;
	std::map<std::string, std::vector<cube_p< real >>> insample;
	insample["input"].resize(1);
    insample["input"][0] = incube_p;

    // run forward and get output
    std::cout<<"run forward..."<<std::endl;
    auto prop = net.forward( std::move(insample) );
    cube<real> out_cube(*prop["output"][0]);
    
    // output size assert
    vec3i outsz( out_cube.shape()[0], out_cube.shape()[1], out_cube.shape()[2] );
#ifdef NDEBUG
	vec3i insz( sz, sy, sx );
	vec3i fov = net.fov();
	assert(outsz == insz - fov + 1);
#endif
    // return ndarray
    std::cout<<"build return ndarray..."<<std::endl;
    return np::from_data(
		out_cube.data(),
		np::dtype::get_builtin<real>(),
		bp::make_tuple(outsz[0],outsz[1],outsz[2]),
		bp::make_tuple(outsz[1]*outsz[2]*sizeof(real), outsz[2]*sizeof(real), sizeof(real)),
		self
	);
}

bp::tuple CNet_fov( bp::object const & self )
{
	network& net = boost::python::extract<network&>(self)();
	vec3i fov_vec =  net.fov();
#ifdef NDEBUG
	std::cout<< "fov (z,y,x): "<<fov_vec[0] <<"x"<< fov_vec[1]<<"x"<<fov_vec[2]<<std::endl;
#endif
	return 	bp::make_tuple(fov_vec[0], fov_vec[1], fov_vec[2]);
}

BOOST_PYTHON_MODULE(pyznn)
{
	Py_Initialize();
	np::initialize();

    bp::class_<network, std::shared_ptr<network>, boost::noncopyable>("CNet",bp::no_init)
        .def("__init__", bp::make_constructor(&CNet_Init))
        .def("set_eta",    	&network::set_eta)
        .def("get_fov",     &CNet_fov)
		.def("forward",     &CNet_forward)
        ;
}
