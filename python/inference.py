#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import numpy as np
import pyznn
import emirt
import time
import matplotlib.pylab as plt
import cPickle as pk
#%% parameters
ftrn = "../dataset/ISBI2012/data/original/train-volume.tif"
flbl = "../dataset/ISBI2012/data/original/train-labels.tif"
fnet_spec = '../networks/srini2d.znn'
fnet = 'net.pickle'
# output size
outsz = np.asarray([1,20,20])
# number of threads
num_threads = 7

#%% load network
net = pk.load( fnet )

# compute inputsize and get input
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1