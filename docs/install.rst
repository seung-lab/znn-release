Installation
============

ZNN Supports Linux and OS X. This guide was developed on Ubuntu 14.04 LTS and OS X Yosemite (10.10.5).

The core of ZNN is written in C++, however we typically control it via a Python interface. We recommend that you follow
the python build instructions as it will result in the interface and a compiled ZNN shared library. The C++ instructions
will generate a binary without an actively developed means of control.


Acquiring a Machine Image -- Recommended
------------------------------------------------------

We have some machine images set up and ready to go for training on:

1. `Amazon Web Services <aws.amazon.com>`_ (called AMIs, Amazon Machine Images)
2. `Google Cloud Platform <cloud.google.com>`_ 

This is the easiest method as the program's dependencies are already loaded and the program is compiled.

Contact `William Wong <william.wong@princeton.edu>`_ to get an AWS account or share the ZNN AMI with your account.

You should find `ZNN` in `/opt/znn-release`. Contact `Jingpeng Wu <jingpeng@princeton.edu>`_ if there is any issue of the AMI. Note that you should run training as `root`. `sudo` is not enough.

Compiling the Python Interface 
------------------------------

To facilitate the usage of ZNN, we have built a python interface. It supports training of boundary and affinity map. Please refer to the `python <https://github.com/seung-lab/znn-release/tree/master/python>`_ folder for further information.

Required Packages
`````````````````

We'll need some libraries for both the C++ core and for Python. For acquiring the python libraries, we recommand using `Anaconda <https://www.continuum.io/downloads>`_, a python distribution that comes with everything.

============================================================== ==================== ============
Library                                                         Ubuntu package name  pip
============================================================== ==================== ============
numpy                                                            python-numpy        numpy
h5py                                                             python-h5py         h5py
matplotlib                                                       python-matplotlib   matplotlib
boost python                                                     libboost-python-dev See below
`Boost.Numpy <http://github.com/ndarray/Boost.NumPy>`_           NA                  See below
`emirt <https://github.com/seung-lab/emirt>`_                    NA                  See below
=============================================================== ==================== ============

We use `Boost.Numpy <http://github.com/ndarray/Boost.NumPy>`_ to facilitate the interaction between python numpy array and the ``cube`` in C++ core. 
To install it, please refer to `Boost.Numpy <http://github.com/ndarray/Boost.NumPy>`_ repository.

Installing Boost.Numpy (OS X)
`````````````````````````````

For convenience, we've provided the following incomplete instructions for OS X:

To install Boost.Numpy you'll need to get boost with Python:

1. Get `Homebrew <https://brew.sh>`_
2. ``brew install boost --with-python``
3. ``brew install boost-python``
4. ``git clone http://github.com/ndarray/Boost.NumPy``
5. ...to be completed. Follow the instructions in the Boost.NumPy repository.


Installing emirt
````````````````

`emirt <https://github.com/seung-lab/emirt>`_ is a home-made python library specially for neuron reconstruction from EM images.

To install it for ZNN, simply run the following command in the folder of ``python``:
::
    git clone https://github.com/seung-lab/emirt.git

If you find it useful and would like to use it in your other programs, you can also install it in a system path (using your PYTHONPATH environment variable).


Compile the core of python interface
````````````````````````````````````
in the folder of ``python/core``:
::
    make -j number_of_cores
  
if you use MKL:
::
    make mkl -j number_of_cores


Compilation of C++ core
-----------------------

The core of ZNN was written with C++ to handle the most computationally expensive forward and backward passes. It is fully functional and can be used to train networks. 

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


Compiling ZNN
-------------

We provide several methods for compilation depending on what tools and libraries you have available to you.


Compiler flags
```````````````

============================== ======================================================================
  Flag                                      Description
============================== ======================================================================
 ZNN_CUBE_POOL                  Use custom memory pool, usually faster
 ZNN_CUBE_POOL_LOCKFREE         Use custom lockfree memory pool, even faster (some memory overhead)
 ZNN_USE_FLOATS                 Use single precision floating point numbers (double precision is default)
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

Intel provides their own optimized C compiler called `icc <https://en.wikipedia.org/wiki/Intel_C%2B%2B_Compiler>`_. If you're interested you might be able to get it and MKL through one of `these packages <https://software.intel.com/en-us/qualify-for-free-software>`_.

in the folder of ``src``:
::
    icc -std=c++1y training_test.cpp -I../../ -I../include -lpthread -lrt -static-intel -DNDEBUG -O3 -mkl=sequential -o training_test

Uninstall ZNN
-------------
Simply remove the ZNN folder. The packages should be uninstalled separately if you would like to.

Resources
---------
- the `travis file <https://github.com/seung-lab/znn-release/blob/master/.travis.yml>`_ shows the step by step installation commands in Ubuntu.
