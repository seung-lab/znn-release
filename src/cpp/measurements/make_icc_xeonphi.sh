#!/bin/bash
module add gcc/4.9
module add intel/compilervars/15
module add intel/mklvars/11.2
icc -mmic -std=c++11 $1.cpp -I../../.. -I../../../src/include -I../../../boost_1_58_0 -DNDEBUG -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -DZNN_USE_MKL_FFT -DZNN_USE_MKL_NATIVE_FFT -DZNN_USE_MKL_DIRECT_CONV -lpthread -lrt -o $1 -mkl=sequential -static-intel -DZNN_DONT_CACHE_FFTS -DZNN_NO_THREAD_LOCAL -DZNN_XEON_PHI
