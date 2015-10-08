#!/bin/bash
g++ -std=c++11 $1.cpp -I../../.. -I../../../src/include -DNDEBUG -g -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -lpthread -lrt -lfftw3f -ljemalloc -o $1 -DZNN_DONT_CACHE_FFTS
