#!/bin/bash
clang++ -std=c++11 $1.cpp -I../../.. -I../../../src/include -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -lpthread  -lfftw3f -ljemalloc -o $1 -DZNN_DONT_CACHE_FFTS
