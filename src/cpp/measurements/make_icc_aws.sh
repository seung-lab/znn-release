#!/bin/bash
/opt/intel/bin/icc \
    -std=c++11 \
    $1.cpp \
    -I../../.. \
    -I../../../src/include \
    -DNDEBUG -O3 \
    -DZNN_CUBE_POOL_LOCKFREE \
    -DZNN_USE_FLOATS \
    -DZNN_USE_MKL_FFT \
    -DZNN_USE_MKL_NATIVE_FFT \
    -DZNN_USE_MKL_DIRECT_CONV \
    -DZNN_DONT_CACHE_FFTS \
    -lpthread -lrt -o $1 -mkl=sequential -static-intel -ljemalloc
