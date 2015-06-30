#!/bin/bash
g++ -std=c++11 measure.cpp ../../../jemalloc-3.6.0/lib/libjemalloc.a -I../../.. -I../../../src/include -DNDEBUG -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -lpthread -lrt -lfftw3f -o sicc -DZNN_DONT_CACHE_FFTS
