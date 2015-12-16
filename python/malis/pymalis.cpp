#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/lexical_cast.hpp>

#include <string>
#include <memory>
#include <cstdint>
#include <assert.h>

#include "flow_graph/computation/zalis.hpp"

#include "core/pyznn_utils.hpp"

namespace bp = boost::python;
namespace np = boost::numpy;

using namespace znn::v4;

template <typename T>
bp::tuple weights2tuple( zalis_weight weights )
{
    std::vector<cube_p<T>>   merger   = weights.merger;
    std::vector<cube_p<T>>   splitter = weights.splitter;
    T re = weights.rand_error;
    T num = weights.num_non_bdr;
    T tp = weights.TP;
    T tn = weights.TN;
    T fp = weights.FP;
    T fn = weights.FN;

    // number of output cubes
    std::size_t sc = merger.size();
    std::size_t sz = merger[0]->shape()[0];
    std::size_t sy = merger[0]->shape()[1];
    std::size_t sx = merger[0]->shape()[2];

    assert( sc==3 );
    // temporal 4D qube pointer
    qube_p<T> tqpm = get_qube<T>( vec4i(sc,sz,sy,sx) );
    qube_p<T> tqps = get_qube<T>( vec4i(sc,sz,sy,sx) );
//    std::cout<<"\n merger: "<<std::endl;
    for (std::size_t c=0; c<sc; c++)
    {
        for (std::size_t k=0; k<sz*sy*sx; k++)
        {
            tqpm->data()[c*sz*sy*sx+k] = merger[c]->data()[k];
            //          std::cout<<merger[c]->data()[k]<<", ";
        }

    }
//    std::cout<<"\n splitter: "<<std::endl;
    for (std::size_t c=0; c<sc; c++)
    {
        for (std::size_t k=0; k<sz*sy*sx; k++)
        {
            tqps->data()[c*sz*sy*sx+k] = splitter[c]->data()[k];
            //std::cout<<splitter[c]->data()[k]<<", ";
        }
    }

    // return ndarray
    np::ndarray arrm = np::from_data( tqpm->data(),
                                     np::dtype::get_builtin<T>(),
                                     bp::make_tuple(sc,sz,sy,sx),
                                     bp::make_tuple(sx*sy*sz*sizeof(T),
                                                    sx*sy*sizeof(T),
                                                    sx*sizeof(T),
                                                    sizeof(T)),
                                     bp::object() );
    np::ndarray arrs = np::from_data( tqps->data(),
                                     np::dtype::get_builtin<T>(),
                                     bp::make_tuple(sc,sz,sy,sx),
                                     bp::make_tuple(sx*sy*sz*sizeof(T),
                                                    sx*sy*sizeof(T),
                                                    sx*sizeof(T),
                                                    sizeof(T)),
                                     bp::object() );

    return bp::make_tuple( arrm.copy(), arrs.copy(),
                           re, num, tp, tn, fp, fn );
}

void show_cubelist( std::vector< cube_p<real> > cl )
{
    // number of output cubes
    std::size_t sc = cl.size();
    std::size_t sz = cl[0]->shape()[0];
    std::size_t sy = cl[0]->shape()[1];
    std::size_t sx = cl[0]->shape()[2];

    for (std::size_t c=0; c < sc; c++)
    {
        std::cout<<"channle: "<<c<<std::endl;
        for (std::size_t k=0; k < sz*sy*sx; k++)
            std::cout<<cl[c]->data()[k]<<", ";
    }
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
/*
    std::cout<<"true affinity map: "<<std::endl;
    show_cubelist( true_affs );
    std::cout<<"affinity map: "<<std::endl;
    show_cubelist( affs );
*/
    // zalis computation
    auto weights = zalis(affs, true_affs, high, low, is_frac_norm);
    return weights2tuple<real>( weights );
}

BOOST_PYTHON_MODULE(pymalis)
{
    Py_Initialize();
    np::initialize();
    def("zalis", pyzalis);
}
