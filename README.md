ZNN
======

[![Build Status](https://travis-ci.org/seung-lab/znn-release.svg?branch=master)](https://travis-ci.org/seung-lab/znn-release)

Most of current deep learning implementation use GPU, but GPU has some limitations:
* SIMD (Single Instruction Multiple Data). A single instruction decoder - all cores do same work.
   * divergence kills performance
* Parallelization done per convolution(s)
    * Direct convolution, computationally expensive
    * FFT, can’t efficiently utilize all cores
* Memory limitations
    * Can’t cache FFT transforms for reuse
    * limit the dense output size (few alternatives for this feature)

ZNN shines when Filter sizes are large so that FFTs are used
* Wide and deep networks
* Bigger output patch
ZNN is the only (reasonable) open source solution
* Very deep networks with large filters
* FFTs of the feature maps and gradients can fit in RAM, but couldn’t fit on the GPU
* run out of the box on future MUUUUULTI core machines

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
