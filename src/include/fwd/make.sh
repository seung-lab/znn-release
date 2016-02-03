g++ $1.cpp -pthread -std=c++11 -I../../.. -I/usr/local/cuda/include -o $1 -lfftw3f   -lcudart -lcudnn  -DNDEBUG -O3 -L/usr/local/cuda/lib64
