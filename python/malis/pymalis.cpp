#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/lexical_cast.hpp>

#include <string>
#include <memory>
#include <cstdint>
#include <assert.h>

#include "computation_graph/computation/zalis.hpp"

#include "pyznn_utils.hpp"

namespace bp = boost::python;
namespace np = boost::numpy;

using namespace znn::v4;


bp::tuple pyzalis( np::ndarray& pyaffs,
                   np::ndarray& pytrue_affs,
                   float high,
                   float low,
                   int is_frac_norm)
{
    // python data structure to c++ data structure
    std::vector< cube_p<real> > true_affs = array2cubelist<real>( pytrue_affs );
    std::vector< cube_p<real> > affs = array2cubelist<real>( pyaffs );

    // zalis computation
    auto weights = zalis(true_affs, affs, high, low, is_frac_norm);

    // transform to python data structure
    np::ndarray pymerger   = cubelist2array<real>(  weights.merger   );
    np::ndarray pysplitter = cubelist2array<real>(  weights.splitter );

    return bp::make_tuple( pymerger, pysplitter );
}

BOOST_PYTHON_MODULE(pymalis)
{
    Py_Initialize();
    np::initialize();

    def("zalis", pyzalis);
}
