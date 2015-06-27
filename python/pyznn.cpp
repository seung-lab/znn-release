#include <boost/python.hpp>
#include "network/parallel/network.hpp"

namespace py = boost::python;
using namespace znn::v4::parallel_network;


class net: network{
public:
    net(    py::list  const & py_ns,
            py::list  const & py_es,
            py::tuple const & py_outsz,
            py::ssize_t py_n_threads )
    {
        // the number of threads
        std::size_t n_threads = py::extract<std::size_t>py_n_threads;
        tm_(n_threads)

        // options
        std::vector<options> ns, es;
        options n,e;
        for (int i=0; i < len(py_ns); i++)
        {
            n = py::extract<options>( py_ns[i] );
            add_nodes(n)
        }
        for (int i=0; i < len(py_es); i++)
        {
            e = py::extract<options>( py_es[i] );
            add_edges(e);
        }

        init( py::extract<vec3i>(py_outsz) );
        create_nodes(ns);
        create_edges(es);
    }
};


BOOST_PYTHON_MODULE(pyznn)
{
    py::class_<net>("net", py::init<py::list, py::list, py::tuple, py::ssize_t>())
        ;
}
