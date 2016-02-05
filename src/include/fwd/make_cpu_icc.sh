icc $1.cpp -O3 -DNDEBUG -DUSE_MKL_FFTS -lrt -std=c++11 -lpthread -mkl=sequential -static-intel -I../../../ -o $1_icc
