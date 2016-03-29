Installation
============

ZNN Supports Linux and OS X. This guide was developed on Ubuntu 14.04 LTS and OS X Yosemite (10.10.5).

Compilation of C++ core
-----------------------

The core of ZNN was written with C++ to handle the most computationally expensive forward and backward pass. It is also fully functional and usable to train networks. 

Required libraries
``````````````````

=============================================================================================== ===================== ===========
Library                                                                                          Ubuntu Package        OS X Homebrew
=============================================================================================== ===================== ===========
`fftw <http://www.fftw.org>`_                                                                    libfftw3-dev          fftw
`boost1.55 <http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.bz2>`_     libboost-all-dev      homebrew/versions/boost155
=============================================================================================== ===================== ===========

Note that fftw is not required when using `intel MKL <https://software.intel.com/en-us/intel-mkl>`_.

For OS X, you can find the above libraries by consulting the table above and using `Homebrew <http://brew.sh/>`_.

Compiler flags
```````````````

============================== ======================================================================
  Flag                                      Description
============================== ======================================================================
 ZNN_CUBE_POOL                  Use custom memory pool, usually faster
 ZNN_CUBE_POOL_LOCKFREE         Use custom lockfree memory pool, even faster (some memory overhead)
 ZNN_USE_FLOATS                 Use single precision floating point numbers
 ZNN_DONT_CACHE_FFTS            Don't cache FFTs for the backward pass
 ZNN_USE_MKL_DIRECT_CONV        Use MKL direct convolution
 ZNN_USE_MKL_FFT                Use MKL fftw wrappers
 ZNN_USE_MKL_NATIVE_FFT         Use MKL native convolution overrides the previous flag
 ZNN_XEON_PHI                   64 byte memory alignment
============================== ====================================================================== 

Compile with make
`````````````````
The easiest way to compile ZNN is to use Makefile.
in the root folder of znn:
::
    make -j number_of_cores
if you use MKL:
::
    make mkl -j number_of_cores

Compile with gcc and clang
``````````````````````````
in the folder of ``src``:
::
    g++ -std=c++1y training_test.cpp -I../../ -I../include -lfftw3 -lfftw3f -lpthread -pthread -O3 -DNDEBUG -o training_test
Notethat g++ should support c++1y standard. v4.8 and later works.

Compile with icc
````````````````
in the folder of ``src``:
::
    icc -std=c++1y training_test.cpp -I../../ -I../include -lpthread -lrt -static-intel -DNDEBUG -O3 -mkl=sequential -o training_test

Python Interface
----------------

To facilitate the usage of ZNN, we have built a python interface. It supports training of boundary and affinity map. Please refer to the `python <https://github.com/seung-lab/znn-release/tree/master/python>`_ folder for further information.

Required Packages
`````````````````

Except the libraries required to build the C++ core, we need some more libraries to build the python interface. For normal python libraries, we recommand to use `Anaconda <https://www.continuum.io/downloads>`_ .

============================================================== ====================
Library                                                         Ubuntu package name
============================================================== ====================
numpy                                                             python-numpy
h5py                                                              python-h5py
matplotlib                                                        python-matplotlib
boost python                                                    libboost-python-dev
`Boost.Numpy <http://github.com/ndarray/Boost.NumPy>`_                  NA
`emirt <https://github.com/seung-lab/emirt>`_                           NA
=============================================================== ====================
We use `Boost.Numpy <http://github.com/ndarray/Boost.NumPy>`_ to facilitate the interaction between python numpy array and the ``cube`` in C++ core. To install it, please refer to `Boost.Numpy <http://github.com/ndarray/Boost.NumPy>`_ repository.

`emirt <https://github.com/seung-lab/emirt>`_ is a home-made python library specially for neuron reconstruction from EM images.

To install it for ZNN, simply run the following command in the folder of ``python``:
::
    git clone https://github.com/seung-lab/emirt.git
If you find it useful and would like to use it in your other programs, you can also install it in a system path (PYTHONPATH).

Compile the core of python interface
````````````````````````````````````
in the folder of ``python/core``:
::
    make -j number_of_cores
  
if you use MKL:
::
    make mkl -j number_of_cores

Uninstall ZNN
-------------
simply remove the ZNN folder. The packages should be uninstalled separately if you would like to.

Resources
---------
- the `travis file <https://github.com/seung-lab/znn-release/blob/master/.travis.yml>`_ shows the step by step installation commands in Ubuntu.
