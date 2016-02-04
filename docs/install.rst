Installation
============

Required libraries
------------------
Supports Linux and MacOS. When using MKL, fftw is not required

=======                                ===================
Library                                Ubuntu package name
=======                                ===================
`fftw<http://www.fftw.org/>`_           libfftw3-dev
`<boost1.55 <http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.bz2>`_                                    libboost-all-dev
`BoostNumpy <http://github.com/ndarray/Boost.NumPy>`_ NA
`jemalloc <http://www.canonware.com/jemalloc/>`_ libjemalloc-dev


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
    g++ -std=c++1y training_test.cpp -I../../ -I../include -lfftw3 -lfftw3f -lpthread -pthread -O3 -DNDEBUG -o training_test

    Compile icc
    -----------
        icc -std=c++1y training_test.cpp -I../../ -I../include -lpthread -lrt -static-intel -DNDEBUG -O3 -mkl=sequential -o training_test

        Python Interface
        ----------------
        To facilitate the usage of ZNN, we have built a python interface. It supports training of boundary and affinity map. Please refer to the `python` folder for further information.
```

Resources
---------
For step by step installation instruction for Linux, please refer to `travis file <https://github.com/seung-lab/znn-release/blob/master/.travis.yml>`_.
