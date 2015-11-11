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


bp::tuple pyzalis( np::ndarray& pytrue_aff,
                   np::ndarray& pyaff,
                   float high,
                   float low,
                   int is_frac_norm)
{
    std::vector< cube_p<real> > true_aff = array2cubelist<real>( pytrue_aff );
    std::vector< cube_p<real> > aff = array2cubelist<real>( pyaff );

    auto weights = zalis(true_aff, aff, high, low, is_frac_norm);

    std::vector< cube_p<real> > merger   = weights.merger;
    std::vector< cube_p<real> > splitter = weights.splitter;

    np::ndarray pymerger   = cubelist2array<real>(  merger   );
    np::ndarray pysplitter = cubelist2array<real>(  splitter );

    return bp::make_tuple( pymerger, pysplitter );
}

BOOST_PYTHON_MODULE(pymalis)
{
    Py_Initialize();
    np::initialize();

    def("zalis", pyzalis);
}
