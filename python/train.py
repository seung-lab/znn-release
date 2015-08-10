#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import pyznn
import emirt
import time
import matplotlib.pylab as plt
import front_end
import cost_fn

#%% parameters
ftrns = list()
flbls = list()
Dir = "/usr/people/jingpeng/seungmount/research/Jingpeng/43_zfish/fish_train/"
ftrns.append( Dir + "Merlin_raw2.tif" )
flbls.append( Dir + "ExportLabels_32bit_Merlin2.tif" )
# network architecture
fnet_spec = '../networks/srini2d.znn'

# mode
dp_type = 'affinity'
# learning rate
eta = 0.01
# momentum
momentum = 0

# output size
outsz = np.asarray([1,20,20])
# number of threads
num_threads = 7

# softmax
is_softmax = False

# rebalance
is_rebalance = False

# malis weight
is_malis = True

# data augmentation
is_data_aug = True

# cost function
cfn = cost_fn.square_loss

# number of iteration per show
Num_iter_per_show = 200
Max_iter = 100000

#%% print parameters
if is_softmax:
    print "using softmax layer"
if is_rebalance:
    print "rebalance the gradients"
if is_malis:
    print "using malis weight"

#%% prepare original input
vol_orgs, lbl_orgs = front_end.read_tifs(ftrns, flbls)

#%% create and initialize the network
print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])
net = pyznn.CNet(fnet_spec, outsz[0],outsz[1],outsz[2],num_threads)
net.set_eta( eta / float(outsz[0] * outsz[1] * outsz[2]) )
net.set_momentum( momentum )

#%% compute inputsize and get input
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1

err = 0;
cls = 0;
err_list = list()
cls_list = list()

# interactive visualization
plt.ion()
plt.show()

start = time.time()
for i in xrange( Max_iter ):
    vol_ins, lbl_outs = front_end.get_sample( vol_orgs, insz, lbl_orgs, outsz, type=dp_type )
    if is_data_aug:
        vol_ins, vol_outs = front_end.data_aug( vol_ins, lbl_outs )
        
    # forward pass
    props = net.forward( vol_ins )

    # softmax
    if is_softmax:
        props = front_end.softmax(props)

    # cost function and accumulate errors
    cerr, ccls, grdts = cfn( props, lbl_outs )
    err = err + cerr
    cls = cls + ccls
    
    # rebalance
    if is_rebalance:
        rb_weights = cost_fn.rebalance( lbl_outs )
        grdts = cost_fn.weight_gradient(grdts, rb_weights)
    if is_malis:
        malis_weights = cost_fn.malis_weights(props)
        grdt_tmp = np.copy( grdts[1] )
        grdts = cost_fn.weight_gradient( grdts, malis_weights )
    
    # run backward pass
    net.backward( grdts )
    
    if i%Num_iter_per_show==0:
        err = err / float(Num_iter_per_show * outsz[0] * outsz[1] * outsz[2])
        cls = cls / float(Num_iter_per_show * outsz[0] * outsz[1] * outsz[2])
        
        err_list.append( err )
        cls_list.append( cls )

        # time
        elapsed = time.time() - start
        print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s"\
                %(i, err, cls, elapsed)
        # real time visualization
        plt.subplot(331),   plt.imshow(vol_ins[0][0,:,:],       interpolation='nearest', cmap='gray')
        plt.xlabel('input')
        plt.subplot(332),   plt.imshow(props[1][0,:,:],    interpolation='nearest', cmap='gray')
        plt.xlabel('inference')
        plt.subplot(333),   plt.imshow(lbl_outs[1][0,:,:], interpolation='nearest', cmap='gray')
        plt.xlabel('lable')
        plt.subplot(334),   plt.imshow(np.log( grdts[1][0,:,:] ),     interpolation='nearest', cmap='gray')
        plt.xlabel('gradient (log)')
        if is_rebalance:
            plt.subplot(335),   plt.imshow(   rb_weights[1][0,:,:],interpolation='nearest', cmap='gray')
            plt.xlabel('rebalance weight')
        if is_malis:
            plt.subplot(335),   plt.imshow(np.log(malis_weights[1][0,:,:]),interpolation='nearest', cmap='gray')
            plt.xlabel('malis weight (log)')
            plt.subplot(336),   plt.imshow( np.abs(grdt_tmp[0,:,:] ),interpolation='nearest', cmap='gray')
            plt.xlabel('gradient befor malis')
        x = np.arange(0, i+1, Num_iter_per_show)
        plt.subplot(337), plt.plot(x, err_list, 'r')
        plt.xlabel('iteration'), plt.ylabel('cost energy')
        plt.subplot(338), plt.plot(x, cls_list, 'b')
        plt.xlabel('iteration'), plt.ylabel( 'classification error' )
            
        plt.pause(1)

        # reset time
        start = time.time()
        # reset err and cls
        err = 0
        cls = 0
