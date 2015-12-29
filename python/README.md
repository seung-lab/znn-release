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
|[tifffile](https://pypi.python.org/pypi/tifffile)|python-tifffile|


Compile
---------------------

* conifgure compile paths in `Makefile`. Modify the pathes in `INC_FLAGS` and `LIB_FLAGS` according to your system library path.
* gcc : `make`
* icc with mkl : `make mkl`

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

Usage
-----
### Train
`python train.py ../experiments/config.cfg`

if you use `python train.py`, the default is `config.cfg` in current folder.

### Forward pass
`python forward.py ../experiments/config.cfg`

### Visualize learning curve
`python zstatistics.py ../experiments/net_statistics.py`

Data format
-----------
The data IO format follows the [standard in Seunglab](https://docs.google.com/spreadsheets/d/1Frn-VH4VatqpwV96BTWSrtMQV0-9ej9soy6HXHgxWtc/edit?usp=sharing).

Contact
-------
* Aleksander Zlateski \<zlateski@mit.edu\>
* Kisuk Lee           \<kisuklee@mit.edu\>
* Jingpeng Wu         \<jingpeng@princeton.edu\>
* Nicholas Turner     \<nturner@cs.princeton.edu\>
