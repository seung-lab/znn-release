Tutorial
========
Since python interface is more convinient to use, this tutorial only focus on usage of python interface.

Data preparation
----------------

Image format
````````````
The dataset is simply a 3D ``tif`` image stacks. 

============== ================= ===========
type            format            bit depth
============== ================= ===========
raw image       .tif              8
label image     .tif              32 or RGB
============== ================= ===========

* For training, you should prepare pairs of ``tif`` files, one is raw image, another is labeled image. 
* For forward pass, only the raw image stack was needed.

Image Configuration
```````````````````
The image pairs was defined as a **Sample**. The image pair of a sample was defined in a dataset configuration file. This `example <https://github.com/seung-lab/znn-release/blob/master/dataset/ISBI2012/dataset.spec>`_ illustrates the meaning of each parameter in the configuration file.

Training
--------




Forward pass
------------

