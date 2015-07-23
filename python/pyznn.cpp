// boost python
#include <Python.h>  // NOLINT(build/include_alpha)
#include <boost/python.hpp>
#include <boost/numpy.hpp>
// system
#include <string>
#include <stdexcept>
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

std::shared_ptr< parallel_network::network > CNetwork_Init(
    std::string net_config_file, std::int64_t outx,
    std::int64_t outy, std::int64_t outz, std::size_t tc )
{
    std::vector<options> nodes;
    std::vector<options> edges;
    parse_net_file(nodes, edges, net_config_file);
    vec3i out_sz(outx, outy, outz);

    // construct the network class
    std::shared_ptr<parallel_network::network> net(
        new parallel_network::network(nodes,edges,out_sz,tc));
    return net;
}

np::ndarray CNetwork_forward( const parallel_network::network& net, const np::ndarray& inarray )
{
    std::map<std::string, std::vector<cube_p<real>>> insample;
    insample["input"].resize(1);
    // input sample volume pointer
    cube<real> in_cube( reinterpret_cast<real*>(inarray.get_data()) );
    insample["input"][0] = std::shared_ptr<cube<real>>(&in_cube);

    auto prop = net.forward( std::move(insample) );

    np::ndarray& outarray;
    outarray.get_data() = *(prop["output"][0]).data()
    return outarray;
}

BOOST_PYTHON_MODULE(pyznn)
{
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    bp::class_<parallel_network::network>, boost::noncopyable>("CNetwork", bp::no_init))
        .def("__init__", bp::make_constructor(&CNetwork_Init))
        .def("_set_eta",    &parallel_network::network::set_eta)
        .def("_fov",        &parallel_network::network::fov)
        .def("forward",   &CNetwork_forward)
        ;
}
