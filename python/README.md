ZNN v4 python interface
======

Required libraries
------------------
Supports Linux and MacOS.

|Library|Ubuntu package name|
|:-----:|-------------------|
|[fftw](http://www.fftw.org/)|libfftw3-dev|
|[boost](http://www.boost.org/)|libboost-all-dev|
|[BoostNumpy](http://github.com/ndarray/Boost.NumPy)|NA|
|[jemalloc](http://www.canonware.com/jemalloc/)|libjemalloc-dev|

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


Compile
---------------------
* gcc : `make`
* icc with mkl : `make mkl`

Contact
-------
* Aleksander Zlateski \<zlateski@mit.edu\>
* Kisuk Lee           \<kisuklee@mit.edu\>
* Jingpeng Wu         \<jingpeng@princeton.edu\>
* Nicholas Turner     \<nturner@cs.princeton.edu\>