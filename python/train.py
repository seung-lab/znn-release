#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import pyznn
import time
import matplotlib.pylab as plt
import front_end
import cost_fn
import ConfigParser


#%% parameters
config = ConfigParser.ConfigParser()
config.read('pyznn.cfg')

num_threads = int( config.get('general', 'num_threads') )
is_softmax = config.getboolean('general', 'is_softmax')

ftrns       = config.get('train', 'ftrns').split(',\n')
flbls       = config.get('train', 'flbls').split(',\n')
fnet_spec   = config.get('train', 'fnet_spec')
dp_type     = config.get('train', 'dp_type')
eta         = config.getfloat('train', 'eta') 
anneal_factor=config.getfloat('train', 'anneal_factor')
momentum    = config.getfloat('train', 'momentum') 
weight_decay= config.getfloat('train', 'weight_decay')
outsz       = np.asarray( [int(x) for x in config.get('train', 'outsz').split(',') ] )
is_data_aug = config.getboolean('train', 'is_data_aug')
is_rebalance= config.getboolean('train', 'is_rebalance')
is_malis    = config.getboolean('train', 'is_malis')
cost_fn_str = config.get('train', 'cost_fn')
Num_iter_per_show = config.getint('train', 'Num_iter_per_show')
Max_iter    = config.getint('train', 'Max_iter')

# cost function
if cost_fn_str == "square_loss":
    cfn = cost_fn.square_loss
elif cost_fn_str == "binomial_cross_entropy":
    cfn = cost_fn.binomial_cross_entropy
elif cost_fn_str == "multinomial_cross_entropy":
    cfn = cost_fn.multinomial_cross_entropy 
else:
    raise NameError('unknown type of cost function')
    

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
eta = eta / float(outsz[0] * outsz[1] * outsz[2])
net.set_eta( eta )
net.set_momentum( momentum )

#%% compute inputsize and get input
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1

err = 0;
cls = 0;
err_list = list()
cls_list = list()
it_list = list()

# interactive visualization
plt.ion()
plt.show()

start = time.time()
for i in xrange(1, Max_iter ):
    vol_ins, lbl_outs = front_end.get_sample( vol_orgs, insz, lbl_orgs, outsz, type=dp_type )
    if is_data_aug:
        vol_ins, vol_outs = front_end.data_aug( vol_ins, lbl_outs )
        
    # forward pass
    props = net.forward( vol_ins )

    # softmax
    if is_softmax:
        props = cost_fn.softmax(props)

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
        # anneal factor
        eta = eta * anneal_factor
        net.set_eta(eta)
        err = err / float(Num_iter_per_show * outsz[0] * outsz[1] * outsz[2])
        cls = cls / float(Num_iter_per_show * outsz[0] * outsz[1] * outsz[2])
        
        err_list.append( err )
        cls_list.append( cls )
        it_list.append( i )

        # time
        elapsed = time.time() - start
        print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s, learning rate: %.4f"\
                %(i, err, cls, elapsed, eta*float(outsz[0] * outsz[1] * outsz[2]) )
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
        
        plt.subplot(337), plt.plot(it_list, err_list, 'r')
        plt.xlabel('iteration'), plt.ylabel('cost energy')
        plt.subplot(338), plt.plot(it_list, cls_list, 'b')
        plt.xlabel('iteration'), plt.ylabel( 'classification error' )
            
        plt.pause(1)

        # reset time
        start = time.time()
        # reset err and cls
        err = 0
        cls = 0
