Introduction
============

ZNN is a multi-core CPU implementation of deep learning for 2D and 3D convolutional networks (ConvNets). While the core is written in C++, it is most often controlled via the Python interface.

When to Use ZNN
---------------

1. Wide and deep networks
2. For bigger output patches ZNN is the only (reasonable) open source solution
3. Very deep networks with large filters
4. FFTs of the feature maps and gradients can fit in RAM, but not on the GPU
5. Runs out of the box on machines with large numbers of cores (e.g. 144+ circa 2016)

ZNN shines when filter sizes are large so that FFTs are used.

CPU vs GPU?
-----------

Most of current deep learning implementations use GPUs, but that approach has some limitations:

1. SIMD (Single Instruction Multiple Data) 
    * GPUs have only a single instruction decoder - all cores do same work. You may have heard that CPUs can also use a variation of SIMD, but they can specify it per core.
    * Branching instructions (if statements) force current GPUs to execute both branches, causing potentially serious decreases in performance.
2. Parallelization done per convolution
    * Direct convolution is computationally expensive
    * FFT can’t efficiently utilize all cores
3. Memory limitations
    * GPUs can’t cache FFT transforms for reuse
    * Limitations on the dense output size (few alternatives for this feature)

What do I need to use ZNN?
--------------------------

Once you've gotten a binary of ZNN either by compiling or using one of our Amazon Web Service AMIs (machine images), here's what you'll need to get started:

1. Image Stacks 
    * Dataset 
    * Ground Truth
    * `tif <https://en.wikipedia.org/wiki/Tagged_Image_File_Format>`_ and `h5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ formats are supported.
2. Sample Definition File (.spec `example <https://github.com/seung-lab/znn-release/blob/master/dataset/ISBI2012/dataset.spec>`_)
    * Provides binding between datasets and ground truths. 
3. Network Architecture File (.znn `example <https://github.com/seung-lab/znn-release/blob/master/networks/srini.znn>`_)
    * Provides layout of your convolutional neural network 
    * Some `sample networks <https://github.com/seung-lab/znn-release/tree/master/networks>`_ are available.
4. Job Configuration File (.cfg `example <https://github.com/seung-lab/znn-release/blob/master/python/config.cfg>`_)
5. Some prior familiarity with convnets. ;)

Keep following this tutorial and you'll learn how to put it all together.

Resources
---------
Tutorial slides: `How to ZNN <https://docs.google.com/presentation/d/1B5g4lgnHN92fD5bkqDCAHraGZL3lz3Df6G-QiYrEWPg/edit?usp=sharing>`_

Publications
------------
* Zlateski, A., Lee, K. & Seung, H. S. (2015) ZNN - A Fast and Scalable Algorithm for Training 3D Convolutional Networks on Multi-Core and Many-Core Shared Memory Machines. (`arXiv link <http://arxiv.org/abs/1510.06706>`_)
* Lee, K., Zlateski, A., Vishwanathan, A. & Seung, H. S. (2015) Recursive Training of 2D-3D Convolutional Networks for Neuronal Boundary Detection. (`arXiv link <http://arxiv.org/abs/1508.04843>`_)
