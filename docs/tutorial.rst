Tutorial
========

Now that you have ZNN set up in an environment you want to use, let's set up an experiment.

Since the python interface is more convenient to use, this tutorial only focuses on it.

1. Importing Experimental Images
-----------------------------

Create a directory called "experiments" in the ZNN root directory. Copy your images to the directory. You'll want to keep track of which images are your source images and which are your ground truth. Make sure you create a training set and a validation set so that you can ensure your training results are meaningful. If you only have one set of images, split them down the middle.

Image format
````````````
The dataset is simply a 3D ``tif`` or ``h5`` image stack. 

============== ================= ===========
type            format            bit depth
============== ================= ===========
raw image       .tif              8
label image     .tif              32 or RGB
============== ================= ===========

* For training, you should prepare pairs of ``tif`` files, one is a stack of raw images, the other is a stack of labeled images. A label is defined as a unique RGBA color.
* For forward pass, only the raw image stack is needed.

Image configuration
```````````````````

Next create a ``.spec`` file that provides the binding between your dataset and ground truth.

The image pairs are defined as a **Sample**. Start with this `example <https://github.com/seung-lab/znn-release/blob/master/dataset/ISBI2012/dataset.spec>`_ and customize it to suit your needs. 

The ``.spec`` file format allows you to specify multiple files as inputs (stack images) and outputs (ground truth labels) for a given experiment. A binding of inputs to outputs is called a sample.

The file structure looks like this, where "N" in imageN can be any positive integer. Items contained inside angle brackets are <option1, option2> etc.

::
    [imageN]
    fnames = path/of/image1
             path/of/image2
    pp_types = <standard2D, none> # preprocess the image by subtracting mean
    is_auto_crop = <yes, no> # crop images to mutually fit and fit ground truth labels

    [labelN]
    fnames = path/of/label1
    pp_types = <one_class, binary_class, affinity, none>
    is_auto_crop = <yes, no>
    fmasks =  path/of/mask1
              path/of/mask1

    [sampleN]
    input = 1 # input image stack of network, the name correpond to the input node name in network configuration file, such as srini.znn
    output = 1 # 

[imageN] options
````````````````
Declaration of source images to train on.

Required:
1. ``fnames``: Paths to image stack files.

Optional:
1. ``pp_types`` (preprocessing types): none (default), standard2D
    standard2D modifies the image by subtracting the mean and dividing by the standard deviation of the pixel values.
2. ``is_auto_crop``: no (default), yes 
    If the corresponding ground truth stack's images are not the same dimension as the image set (e.g. image A is 1000px x 1000pxand label A is 100px x 100px), then the smaller image will be centered in the larger image and the larger image will be cropped around it.


[labelN] options
````````````````
Declaration of ground truth labels to evaluate training on.

Required:
1. ``fnames``: Paths to label stack files.

Optional:
1. ``pp_types`` (preprocessing types): none (default), one_class, binary_class, affinity

==================== =========================================================
 Preprocessing Type  Function
==================== =========================================================
 none                Don't do anything.
 one_class           Normalize values, threshold at 0.5
 binary_class        one_class + generate extra inverted version
 affinity            Generate X, Y, & Z stacks for training on different axes   
==================== =========================================================

2. ``is_auto_crop``: no (default), yes 
    If the corresponding ground truth stack's images are not the same dimension as the image set (e.g. image A is 1000px x 1000pxand label A is 100px x 100px), then the smaller image will be centered in the larger image and the larger image will be cropped around it.

3. ``fmasks``: Paths to mask files
    fmasks are used like cosmetics to coverup damaged parts of images so that your neural net
    doesn't learn useless information. White is on, black is off. The same file types are supported as for regular images.

[sampleN] options
`````````````````

Declaration of binding between images and labels. You'll use the sample number in your training configuration to decide which image sets to train on.

Required:
1. input: (int > 0) should correspond to the N in an [imageN]. e.g. ``input: 1`` 
2. output: (int > 0) should correspond to the N in a [labelN]. e.g. ``output: 1``


2. Network Architecture Configuration
----------------------------------

Please refer to the `examples <https://github.com/seung-lab/znn-release/tree/master/networks>`_.

3. Training
-----------

Parameter configuration
```````````````````````
The training/forward parameters can be set using a configuration file. This `example <https://github.com/seung-lab/znn-release/blob/master/python/config.cfg>`_ illustrates the parameters and their meaning in the comments.

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

Forward Pass
------------
run the following command:
::
    python forward.py -c path/of/config.cfg
