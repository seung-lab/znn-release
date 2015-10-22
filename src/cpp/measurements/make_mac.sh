#!/bin/bash
clang++ -std=c++11 $1.cpp -I../../.. -I../../../src/include -DNDEBUG -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -lpthread -lfftw3f -o $1  -ljemalloc
