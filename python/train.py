#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import pyznn
import emirt
import time
# parameters
ftrn = "../dataset/ISBI2012/data/original/train-volume.tif"
flbl = "../dataset/ISBI2012/data/original/train-labels.tif"
fnet_spec = '../networks/srini2d.znn'
# learning rate
eta = 0.01
# output size
outsz = np.asarray([1,5,5])
# number of threads
num_threads = 7

# prepare input
vol = emirt.io.imread(ftrn).astype('float32')
lbl = emirt.io.imread(flbl).astype('float32')
# normalize the training volume
vol = emirt.volume_util.norm( vol )
lbl = (lbl>0.5).astype('float32')

print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])
net = pyznn.CNet(fnet_spec, outsz[0],outsz[1],outsz[2],num_threads)
net.set_eta( eta / float(outsz[0] * outsz[1] * outsz[2]) )

# compute inputsize and get input
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1

err = 0;
cls = 0;
# get gradient
from front_end import get_sample
from cost_fn import square_loss
for i in xrange(1,1000000):
    vol_in, lbl_out = get_sample( vol, insz, lbl, outsz )
    
    # forward pass
    prop = net.forward(vol_in.astype('float32'))
        
    cerr, ccls, grdt = square_loss( prop, lbl_out )  
    err = err + cerr
    cls = cls + ccls  
    
    # run backward pass
    net.backward(grdt)
    
    if i%1000==0:
        err = err / float(1000 * outsz[0] * outsz[1] * outsz[2]) 
        cls = cls / float(1000 * outsz[0] * outsz[1] * outsz[2])
        print "iteration %d,    sqerr: %.4f,    clserr: %.4f"%(i, err, cls)
        err = 0
        cls = 0
        
#%% visualization
com = emirt.show.CompareVol((vol_in, lbl_out))
com.vol_compare_slice()