znn-release
===========
Multi-core CPU implementation of deep learning for 2D and 3D convolutional networks (ConvNets).



Required libraries
------------------
Currently we only support linux environments.

|Library|Ubuntu package name|
|:-----:|-------------------|
|[fftw](http://www.fftw.org/)|libfftw-dev|
|[boost](http://www.boost.org/)|libbost-all-dev|



Compile & clean
---------------
    make
    make clean

If compile is successful, an executalbe named **znn** will be generated under the directory [bin](./bin/).



Directories
-----------
### [bin](./bin/)
An executable will be generated here.

### [matlab](./matlab/)
Matlab functions for preparing training data and analyzing training results.

### [src](./src/)

### [zi](./zi/) 
General purpose C++ library, written and maintained by Aleksander Zlateski.



Contact
-------
* Aleksander Zlateski \<zlateski@mit.edu\>
* Kisuk Lee \<kisuklee@mit.edu\>
