Tutorial
========
Since the python interface is more convenient to use, this tutorial only focuses on it.

File preparation
----------------

Image format
````````````
The dataset is simply a 3D ``tif`` image stack. 

============== ================= ===========
type            format            bit depth
============== ================= ===========
raw image       .tif              8
label image     .tif              32 or RGB
============== ================= ===========

* For training, you should prepare pairs of ``tif`` files, one is a stack of raw images, the other is a stack of labeled images. 
* For forward pass, only the raw image stack was needed.

Image configuration
```````````````````
The image pairs are defined as a **Sample**. The image pairs in a sample are defined in a dataset configuration file. This `example <https://github.com/seung-lab/znn-release/blob/master/dataset/ISBI2012/dataset.spec>`_ illustrates the meaning of each parameter in the configuration file.

Network architecture configuration
``````````````````````````````````
Please refer to the `examples <https://github.com/seung-lab/znn-release/tree/master/networks>`_.

Parameter configuration
```````````````````````
The training/forward parameters can be set using a configuration file. This `example <https://github.com/seung-lab/znn-release/blob/master/python/config.cfg>`_ illustrates the parameters and their meaning in the comments.

Training
--------

Run a training
``````````````
After setting up the configuration file, you can run a training: 
::
    python train.py -c path/of/config.cfg 

Resume a training
`````````````````
Since the network is periodically saved, we can resume training whenever we want to. By default, ZNN will automatically resume the latest training net (``net_current.h5``) in a folder, which was specified by the ``train_net`` parameter in the configuration file. 

To resume training a specific network, we can use the seeding function:
::
    python train.py -c path/of/config.cfg -s path/of/seed.h5

Transfer learning
`````````````````
Sometimes, we would like to utilize a trained network. If the network architectures of trained and initialized network are the same, we call it ``Loading``. Otherwise, we call it ``Seeding``, in which case the trained net is used as a seed to initialize part of the new network. Our implementation merges ``Loading`` and ``Seeding``. Just use the synonymous ``-s`` or ``--seed`` command line flags. 
::
    python train.py -c path/of/config.cfg -s path/of/seed.h5

Forward pass
------------
run the following command:
::
    python forward.py -c path/of/config.cfg
