### emirt
EM Image Reconstruction Toolbox

EMIRT is a python library for 3D image processing and analysis. It was built for general usage in [Seunglab](http://seunglab.org/) for EM image segmentation and neuron reconstruction.

Required libraries
---------

|Library|Ubuntu package name|
|:-----:|-------------------|
|tifffile|python-tifffile|
|h5py|python-h5py|
|numpy|python-numpy|
|matplotlib|python-matplotlib|
|scipy|python-scipy|

Install
--------
* `cd` to a library folder and download `emirt`. git clone https://github.com/seung-lab/emirt.git
* add the following line to the end of ~/.bashrc (~/.bashprofile in Mac OS) `export PYTHONPATH=$PYTHONPATH:"folder/of/emirt"`
* run: `source ~/.bashrc`
