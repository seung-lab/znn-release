#!/bin/bash
/opt/intel/bin/icc -std=c++11 measure.cpp -I../../.. -I../../../src/include -DNDEBUG -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -DZNN_USE_MKL_FFT -DZNN_USE_MKL_NATIVE_FFT -DZNN_USE_MKL_DIRECT_CONV -lpthread -lrt -o sicc -mkl=sequential -static-intel -DZNN_DONT_CACHE_FFTS -ljemalloc
