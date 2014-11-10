znn-release
===========
Multi-core CPU implementation of deep learning for 2D and 3D convolutional networks (ConvNets).



Required libraries
------------------
Currently we only support linux environments.

|Library|Ubuntu package name|
|:-----:|-------------------|
|[fftw](http://www.fftw.org/)|libfftw3-dev|
|[boost](http://www.boost.org/)|libboost-all-dev|



Compile & clean
---------------
    make
    make clean

If compile is successful, an executalbe named **znn** will be generated under the directory [bin](./bin/).



Directories
-----------
#### [bin](./bin/)
An executable will be generated here.

#### [matlab](./matlab/)
Matlab functions for preparing training data and analyzing training results.

#### [src](./src/)
C++ source code.
* [core](./src/core) -- core classes for constructing ConvNets and performing multi-core parallelized computations.
* [cost_fn](./src/cost_fn/) -- cost (objective) functions for training ConvNets.
* [error_fn](./src/error_fn/) -- linear and/or non-linear activation functions for neurons.
* [front_end](./src/front_end/) -- an interface for specifying ConvNet architecure, training data, and training options.
* [initializer](./src/initializer/) -- random initializers for weights of ConvNets.

#### [zi](./zi/) 
General purpose C++ library, written and maintained by Aleksander Zlateski.



Slides
------
* [Core Architecture](https://docs.google.com/presentation/d/1O1Xkyx71eUAZnNa784wvEKzqF7-lon7qSbAWZGRKg4E/edit)
* [How to Tutorial](https://docs.google.com/presentation/d/1yPQ5xDkhHeyfL7Xt4TvoEiJzHkpoas-as5nWO48hjrQ/edit)


Contact
-------
* Aleksander Zlateski \<zlateski@mit.edu\>
* Kisuk Lee \<kisuklee@mit.edu\>
