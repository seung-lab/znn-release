// boost python
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>  // NOLINT(build/include_alpha)
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <numpy/ndarrayobject.h>

// system
#include <string>
#include <memory>
#include <cstdint>
// znn
#include "network/parallel/network.hpp"
#include "cube/cube.hpp"
#include "types.hpp"
#include "options/options.hpp"
#include "network/trivial/trivial_fft_network.hpp"
#include <zi/zargs/zargs.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;
using namespace znn::v4;
using namespace znn::v4::parallel_network;

//#ifdef ZNN_USE_FLOATS
//typedef NPY_FLOAT32		NPY_DTYPE;
//#else
//typedef NPY_DOUBLE		NPY_DTYPE;
//#endif

std::shared_ptr< network > CNetwork_Init(
    std::string net_config_file, std::int64_t outx,
    std::int64_t outy, std::int64_t outz, std::size_t tc )
{
    std::vector<options> nodes;
    std::vector<options> edges;
    parse_net_file(nodes, edges, net_config_file);
    vec3i out_sz(outx, outy, outz);

    // construct the network class
    std::shared_ptr<network> net(
        new network(nodes,edges,out_sz,tc));
    return net;
}

cube_p<real> ndarray2cube_p( const np::ndarray& inarray,
		std::size_t sz,
		std::size_t sy,
		std::size_t sx)
{
	vec3i size(sx,sy,sz);
	boost::multi_array_ref<real,3> cube_ref( reinterpret_cast<real*>(inarray.get_data()), extents[sx][sy][sz] );
	cube<real> cube( cube_ref );
	cube_p<real> cube_p = std::shared_ptr< cube<real> >( &cube );
	return cube_p;
}

np::ndarray CNetwork_forward( bp::object const & self, network& net, const np::ndarray& inarray )
{
	// volume size
	std::size_t sz = inarray.shape(0);
	std::size_t sy = inarray.shape(1);
	std::size_t sx = inarray.shape(2);

	// setup input volume
    cube_p<real> incube_p = ndarray2cube_p( inarray, sz,sy,sx );

	std::map<std::string, std::vector<cube_p< real >>> insample;
	insample["input"].resize(1);
    insample["input"][0] = incube_p;

    // run forward and get output
    auto prop = net.forward( std::move(insample) );
    cube<real> out_cube(*prop["output"][0]);

    // create a PyObject * from pointer and data to return
    return np::from_data(
		out_cube.data(),
		np::dtype::get_builtin<real>(),
		bp::make_tuple(sz,sy,sx),
		bp::make_tuple(sy*sx*sizeof(real), sx*sizeof(real), sizeof(real))
		self
	);
}

//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CNetwork_forward_overloads, CNetwork_forward, 2, 2)

BOOST_PYTHON_MODULE(pyznn)
{
//    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	Py_Initialize();
	// Initialize NumPy
	np::initialize();

    bp::class_<network, std::shared_ptr<network>>("CNetwork", bp::no_init)
        .def("__init__", bp::make_constructor(&CNetwork_Init))
        .def("_set_eta",    &network::set_eta)
        .def("_fov",        &network::fov)
		.def("forward",     &CNetwork_forward)
        ;
}
