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

template <typename T>
bp::tuple weights2tuple( zalis_weight weights )
{
    std::vector<cube_p<T>>   merger   = weights.merger;
    std::vector<cube_p<T>>   splitter = weights.splitter;
    T re = weights.rand_error;

    // number of output cubes
    std::size_t sc = merger.size();
    std::size_t sz = merger[0]->shape()[0];
    std::size_t sy = merger[0]->shape()[1];
    std::size_t sx = merger[0]->shape()[2];

    assert( sc==3 );
    // temporal 4D qube pointer
    qube_p<T> tqp = get_qube<T>( vec4i(sc*2,sz,sy,sx) );
    for (std::size_t c=0; c<sc; c++)
    {
        for (std::size_t k=0; k<sz*sy*sx; k++)
            tqp->data()[c*sz*sy*sx+k] = merger[c]->data()[k];
    }

    for (std::size_t c=sc; c<2*sc; c++)
    {
        for (std::size_t k=0; k<sz*sy*sx; k++)
            tqp->data()[c*sz*sy*sx+k] = splitter[c-sc]->data()[k];
    }

    // return ndarray
    np::ndarray arr = np::from_data( tqp->data(),
                                     np::dtype::get_builtin<T>(),
                                     bp::make_tuple(2*sc,sz,sy,sx),
                                     bp::make_tuple(sx*sy*sz*sizeof(T),
                                                    sx*sy*sizeof(T),
                                                    sx*sizeof(T),
                                                    sizeof(T)),
                                     bp::object() );
    return bp::make_tuple( arr, re );
}

bp::tuple pyzalis( np::ndarray& pyaffs,
                   np::ndarray& pytrue_affs,
                   float high,
                   float low,
                   std::size_t is_frac_norm)
{
    // python data structure to c++ data structure
    std::vector< cube_p<real> > true_affs = array2cubelist<real>( pytrue_affs );
    std::vector< cube_p<real> > affs = array2cubelist<real>( pyaffs );

    // zalis computation
    auto weights = zalis(true_affs, affs, high, low, is_frac_norm);

    return weights2tuple<real>( weights );
}

BOOST_PYTHON_MODULE(pymalis)
{
    Py_Initialize();
    np::initialize();
    def("zalis", pyzalis);
}
