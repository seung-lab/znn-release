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
//namespace z4 = znn::v4;
using namespace znn::v4;
using namespace znn::v4::parallel_network;

//#ifdef ZNN_USE_FLOATS
//	typedef NPY_FLOAT32		NPY_DTYPE;
//#else
//	// here has a bug!!!!!!!!, should use float64, but can not compile!
//	typedef NPY_FLOAT32		NPY_DTYPE;
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

template <typename T>
bp::numeric::array CNetwork_forward( network& net, const bp::numeric::array& inarray )
{
    std::map<std::string, std::vector<cube_p<T>>> insample;
    insample["input"].resize(1);
    // input sample volume pointer
    cube<T> in_cube( reinterpret_cast<T*>(inarray.ptr()) );
    insample["input"][0] = std::shared_ptr<cube<T>>(&in_cube);

    auto prop = net.forward( std::move(insample) );
    cube<T> out_cube(*prop["output"][0]);

    // create a PyObject * from pointer and data
    npy_intp size = out_cube.shape()[2] * out_cube.shape()[1] *	out_cube.shape()[0] ;
    PyObject * pyObj = PyArray_SimpleNewFromData(3, &size, NPY_DOUBLE, out_cube.data());
    bp::handle<> handle( pyObj );
    bp::numeric::array outarray( handle );

    return outarray;
}

BOOST_PYTHON_MODULE(pyznn)
{
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    bp::class_<network, std::shared_ptr<network>>("CNetwork", bp::no_init)
        .def("__init__", bp::make_constructor(&CNetwork_Init))
        .def("_set_eta",    &network::set_eta)
        .def("_fov",        &network::fov)
        .def("forward",     &CNetwork_forward)
        ;
}
