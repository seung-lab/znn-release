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
    input = 1
    output = 1  

[imageN] options
````````````````
Declaration of source images to train on.

Required:

1. ``fnames``: Paths to image stack files.

Optional:

1. ``pp_types`` (preprocessing types): none (default), standard2D
    standard2D modifies the image by subtracting the mean and dividing by the standard deviation of the pixel values.
2. ``is_auto_crop``: no (default), yes 
    If the corresponding ground truth stack's images are not the same dimension as the image set (e.g. image A is 1000px x 1000px and label A is 100px x 100px), then the smaller image will be centered in the larger image and the larger image will be cropped around it.


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
    If the corresponding ground truth stack's images are not the same dimension as the image set (e.g. image A is 1000px x 1000px and label A is 100px x 100px), then the smaller image will be centered in the larger image and the larger image will be cropped around it.

3. ``fmasks``: Paths to mask files
    fmasks are used like cosmetics to coverup damaged parts of images so that your neural net
    doesn't learn useless information. Pixel values greater than zero are on. That is to say, white is on, black is off. The same file types are supported as for regular images.

[sampleN] options
`````````````````

Declaration of binding between images and labels. You'll use the sample number in your training configuration to decide which image sets to train on.

Required:

1. ``input``: (int > 0) should correspond to the N in an [imageN]. e.g. ``input: 1`` 
2. ``output``: (int > 0) should correspond to the N in a [labelN]. e.g. ``output: 1``


2. Network Architecture Configuration
-------------------------------------

We have a custom file format ``.znn`` for specifying the layout of your neural network. It works based on a few simple concepts. 

1. Each of the input nodes of the network represent an image stack.
2. The network consists of layers whose size can be individually specified. 
3. The edge betwen the layers specify not only the data transfer from one layer to another (e.g. one to one, or fully connected), they also prescribe a transformation, e.g. a filter or weight, to be applied. 
4. After all the weights or filters have been applied, the inputs are summed and a pixel-wise transfer function (e.g. a `sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function>`_ or `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_) is applied.
5. The type of the edges determines if the layers its connecting is a one-to-one mapping or is fully connected. For example, a convolution type will result in fully connected layers.
6. The output layer represents whatever you're training the network to do. One common output is the predicted labels for an image stack as a single node.

You can find example network N4 `here <https://github.com/seung-lab/znn-release/blob/master/networks/N4.znn>`_.

Here's an example excepted from the N4 network:
::
    nodes input
    type input
    size 1

    edges conv1
    type conv
    init xavier
    size 1,4,4
    stride 1,1,1
    input input
    output nconv1

    nodes nconv1
    type transfer
    function rectify_linear
    size 48

    edges pool1
    type max_filter
    size 1,2,2
    stride 1,2,2
    input nconv1
    output npool1

    nodes npool1
    type sum
    size 48

    ....

    edges conv6
    type conv
    init xavier
    size 1,1,1
    stride 1,1,1
    input nconv5
    output output

    nodes output
    type transfer
    function linear
    size 2


The ``.znn`` file is comprised of two primary objects -- nodes and edges. An object declaration consists of the type ``nodes`` or ``edges`` followed by its name on a new line followed by its parameters.

``nodes`` type declaration
``````````````````````````

Note: In the Description column for functions, the relevant funciton_args are presented as:
``[ comma,seperated,variables | default,values,here ]``

================ =========== =================== ================================================================
 Property         Required    Options             Description                                                    
================ =========== =================== ================================================================
 nodes            Y           $NAME               Symbolic identifier for other layers to reference. The names "input" and "output" are special and represent the input and output layers of the entire network.
 type             Y           sum                 Perform a simple weighted summing of the inputs to this node.
 ..               ..          transfer            Perform a summation of the input nodes and then apply a transfer function (c.f. function).
 function         N           linear              Line. [ slope,intercept | 1,1 ]
 ..               ..          rectify_linear      Rectified Linear Unit (ReLU)
 ..               ..          tanh                Hyperbolic Tangent. [ amplitude,frequency | 1,1 ]
 ..               ..          soft_sign           x / (1 + abs(x))
 ..               ..          logistics           Logistic function aka sigmoid. Has gradient.
 ..               ..          forward_logistics   Same as "logistics" but without a gradient?
 function_args    N           $VALUES             Input comma seperated values of the type appropriate for the selected function.
 size             Y           $POSTIVE_INTEGER    The number of nodes in this layer.
================ =========== =================== ================================================================

``edges`` type declaration
``````````````````````````

Note: In the Description column for functions, the relevant init_args are presented as:
``[ comma,seperated,variables | default,values,here ]``

================ =========== =================== ================================================================
 Property         Required    Options             Description                                                    
================ =========== =================== ================================================================
 edges            Y           $NAME               Symbolic identifier for other layers to reference
 type             Y           conv                Layers are fully connected and convolution is applied.
 ..                           max_filter          Layers are connected one-to-one and max filtering is applied.
 init             Y           zero                Filters are zeroed out.
 ..                           constant            Filters are set to a particular constant. ``[ constant | ? ]``
 ..                           uniform             Filters are uniformly randomly initialized. ``[ min,max | -0.1,0.1 ]``
 ..                           gaussian            Filters are gaussian randomly initialized. ``[ mean,stddev | 0,0.01 ]``
 ..                           bernoulli           Filters are bernoulli randomly initialized. ``[ p | 0.5 ]``
 ..                           xavier              Filters are assigned as described in `Glorot and Bengio 2010 <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`_ [1].
 ..                           msra                Filters are assigned as described in `He, Zhang, Ren and Sun 2015 <http://arxiv.org/abs/1502.01852>`_ [2].
 init_args        N           $VALUES             Input comma seperated values of the type appropriate for the selected init.
 size             Y           $X,$Y,$Z            Size of sliding window in pixels. 2D nets can be implemented by setting $Z to 1.
 stride           Y           $X,$Y,$Z            How far to jump in each direction in pixels when sliding the window.
 input            Y           $NODES_NAME         Name of source ``nodes`` layer that the edge will be transforming.
 output           Y           $NODES_NAME         Name of destination ``nodes`` layer that the edge will be transforming.
================ =========== =================== ================================================================

[1] Glorot and Bengio. "Understanding the difficulty of training deep feedforward neural networks". JMLR 2010. http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

[2] He, Zhang, Ren and Sun. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" CVPR 2015. http://arxiv.org/abs/1502.01852
 

For more examples, please refer to the `networks <https://github.com/seung-lab/znn-release/tree/master/networks>`_ directory.

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
