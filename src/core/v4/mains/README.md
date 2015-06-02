ZNN v4
======

Required libraries
------------------
Supports Linux and MacOS. When using MKL fftw is not required

|Library|Ubuntu package name|
|:-----:|-------------------|
|[fftw](http://www.fftw.org/)|libfftw3-dev|
|[boost](http://www.boost.org/)|libboost-all-dev|

Compiler flags
--------------

|Flag|Description|
|:-----:|-------------------|
|ZNN_CUBE_POOL|Use custom memory pool, usually faster|
|ZNN_CUBE_POOL_LOCKFREE|Use custom lockfree memory pool, even faster (some memory overhead)|
|ZNN_USE_FLOATS|Use single precision floating point numbers|
|ZNN_DONT_CACHE_FFTS|Don't cache FFTs for the backward pass|
|ZNN_USE_MKL_DIRECT_CONV|Use MKL direct convolution|
|ZNN_USE_MKL_FFT|Use MKL fftw wrappers|
|ZNN_USE_MKL_NATIVE_FFT|Use MKL native convolution overrides the previous flag|
|ZNN_XEON_PHI|64 byte memory alignment|


Compile gcc and clang
---------------------
    g++ -std=c++1y training_test.cpp -I../../../.. -I.. -lfftw3 -lfftw3f -lpthread -O3 -DNDEBUG -o training_test

Compile icc
-----------
    icc -std=c++1y training_test.cpp -I../../../.. -I.. -lpthread -lrt -static-intel -DNDEBUG -O3 -mkl=sequential -o training_test


Contact
-------
* Aleksander Zlateski \<zlateski@mit.edu\>