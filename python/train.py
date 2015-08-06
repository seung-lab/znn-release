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
ftrn = "/usr/people/jingpeng/seungmount/research/Jingpeng/43_zfish/fish_train/Merlin_raw2.tif"
flbl = "/usr/people/jingpeng/seungmount/research/Jingpeng/43_zfish/fish_train/ExportLabels_32bit_Merlin2.tif"
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

# cost function
cfn = cost_fn.square_loss

# number of iteration per show
Num_iter_per_show = 100

#%% print parameters
if is_softmax:
    print "using softmax layer"
if is_rebalance:
    print "rebalance the gradients"
if is_malis:
    print "using malis weight"

#%% prepare original input
vol_org = emirt.io.imread(ftrn).astype('float32')
lbl_org = emirt.io.imread(flbl).astype('float32')
# normalize the training volume
vol_org = vol_org / 255

#%% create and initialize the network
print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])
net = pyznn.CNet(fnet_spec, outsz[0],outsz[1],outsz[2],num_threads)
net.set_eta( eta / float(outsz[0] * outsz[1] * outsz[2]) )
net.set_momentum( momentum )

# compute inputsize and get input
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1

err = 0;
cls = 0;
# get gradient
plt.ion()
plt.show()

start = time.time()
for i in xrange(1,1000000):
    vol_in, lbl_outs = front_end.get_sample( vol_org, insz, lbl_org, outsz, type=dp_type )
    inputs = list()
    inputs.append( np.ascontiguousarray(vol_in) )
    # forward pass
    props = net.forward( inputs )

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

    if i%Num_iter_per_show==0:
        err = err / float(Num_iter_per_show * outsz[0] * outsz[1] * outsz[2])
        cls = cls / float(Num_iter_per_show * outsz[0] * outsz[1] * outsz[2])

        # time
        elapsed = time.time() - start
        print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s"\
                %(i, err, cls, elapsed)
        # real time visualization
        plt.subplot(321),   plt.imshow(vol_in[0,:,:],       interpolation='nearest', cmap='gray')
        plt.xlabel('input')
        plt.subplot(322),   plt.imshow(prop[1][0,:,:],    interpolation='nearest', cmap='gray')
        plt.xlabel('inference')
        plt.subplot(323),   plt.imshow(lbl_outs[1][0,:,:], interpolation='nearest', cmap='gray')
        plt.xlabel('lable')
        plt.subplot(324),   plt.imshow(np.log( grdts[1][0,:,:] ),     interpolation='nearest', cmap='gray')
        plt.xlabel('gradient (log)')
        if is_rebalance:
            plt.subplot(325),   plt.imshow(   rb_weights[1][0,:,:],interpolation='nearest', cmap='gray')
            plt.xlabel('rebalance weight')
        if is_malis:
            plt.subplot(325),   plt.imshow(np.log(malis_weights[1][0,:,:]),interpolation='nearest', cmap='gray')
            plt.xlabel('malis weight (log)')
            plt.subplot(326),   plt.imshow( np.abs(grdt_tmp[0,:,:] ),interpolation='nearest', cmap='gray')
            plt.xlabel('gradient befor malis')
            
        plt.pause(3)

        # reset time
        start = time.time()
        # reset err and cls
        err = 0
        cls = 0

    # run backward pass
    net.backward( grdts )


#%% visualization
#com = emirt.show.CompareVol((vol_in, lbl_out))
#com.vol_compare_slice()
