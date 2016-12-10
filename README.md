ZNN
======

[![Build Status](https://travis-ci.org/seung-lab/znn-release.svg?branch=master)](https://travis-ci.org/seung-lab/znn-release)

ZNN is a multicore CPU implementation of deep learning for sliding window convolutional networks.  This type of ConvNet is used for image-to-image transformations, i.e., it produces dense output. Use of ZNN is currently deprecated, except for the special case of 3D kernels that are 5x5x5 or larger. In this case, the FFT-based convolutions of ZNN are able to compensate for the lower FLOPS of most CPUs relative to GPUs.

For most dense output ConvNet applications, we are currently using https://github.com/torms3/DataProvider with Caffe or TensorFlow.

ZNN will be superseded by ZNNphi, which is currently in preparation. By vectorizing direct convolutions, ZNNphi more efficiently utilizes the FLOPS of multicore CPUs for 2D and small 3D kernels. This includes manycore CPUs like Xeon Phi Knights Landing, which has narrowed the FLOPS gap with GPUs.

Resources
---------
* [**Documentation**](http://znn-release.readthedocs.org/en/latest/index.html#)
* [**Slides: How to ZNN**](https://docs.google.com/presentation/d/1B5g4lgnHN92fD5bkqDCAHraGZL3lz3Df6G-QiYrEWPg/edit?usp=sharing)

Publications
------------
* Zlateski, A., Lee, K. & Seung, H. S. (2015) ZNN - A Fast and Scalable Algorithm for Training 3D Convolutional Networks on Multi-Core and Many-Core Shared Memory Machines. ([arXiv link](http://arxiv.org/abs/1510.06706))
* Lee, K., Zlateski, A., Vishwanathan, A. & Seung, H. S. (2015) Recursive Training of 2D-3D Convolutional Networks for Neuronal Boundary Detection. ([arXiv link](http://arxiv.org/abs/1508.04843))

Contact
-------
C++ core
* Aleksander Zlateski \<zlateski@mit.edu\>
* Kisuk Lee \<kisuklee@mit.edu\>

Python Interface
* Jingpeng Wu \<jingpeng@princeton.edu\>
* Nicholas Turner \<nturner@cs.princeton.edu\>
