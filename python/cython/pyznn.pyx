import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair   cimport pair
from libcpp.list   cimport list

cdef extern from "options.hpp" namespace "znn::v4":
    cdef cppclass options:
        options()
        options( list[ pair[string,string] ] )

cdef extern from "network.hpp" namespace "znn::v4::parallel_network":
    cdef cppclass network:
        network( vector[options], vector[options], vec3i, size_t n_threads) except +
        set_eta( real )
        set_momentum( real )
        set_weight_decay( real )
