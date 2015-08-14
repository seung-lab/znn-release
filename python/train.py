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
config.read('config.cfg')

fnet_spec   = config.get('general', 'fnet_spec')
num_threads = int( config.get('general', 'num_threads') )
is_softmax = config.getboolean('general', 'is_softmax')
dp_type     = config.get('general', 'dp_type')

ftrns       = config.get('train', 'ftrns').split(',\n')
flbls       = config.get('train', 'flbls').split(',\n')
eta         = config.getfloat('train', 'eta') 
anneal_factor=config.getfloat('train', 'anneal_factor')
momentum    = config.getfloat('train', 'momentum') 
weight_decay= config.getfloat('train', 'weight_decay')
outsz       = np.asarray( [x for x in config.get('train', 'outsz').split(',') ], dtype=np.int64 )
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
net = pyznn.CNet(fnet_spec, outsz, num_threads)
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
    vol_ins, lbl_outs = front_end.get_sample( vol_orgs, insz, lbl_orgs, outsz, dp_type=dp_type )
    if is_data_aug:
        vol_ins, lbl_outs = front_end.data_aug( vol_ins, lbl_outs )
        
    # forward pass
    props = net.forward( np.ascontiguousarray(vol_ins) ).astype('float32')

    # softmax
    if is_softmax:
        props = cost_fn.softmax(props)
    
    # cost function and accumulate errors
    cerr, grdts = cfn( props.astype('float32'), lbl_outs.astype('float32') )
    ccls = np.count_nonzero( (props>0.5)!= lbl_outs )
    err = err + cerr
    cls = cls + ccls
    
    # rebalance
    if is_rebalance:
        rb_weights = cost_fn.rebalance( lbl_outs )
        grdts = grdts * rb_weights
    if is_malis:
        grdts_bm = np.copy(grdts)
        malis_weights = cost_fn.malis_weights(props)
        grdts = grdts * malis_weights 
    
    # run backward pass
    net.backward( np.ascontiguousarray(grdts) )
    
    if i%Num_iter_per_show==0:
        # anneal factor
        eta = eta * anneal_factor
        net.set_eta(eta)
        # normalize
        err = err / float(Num_iter_per_show * props.shape[0] * outsz[0] * outsz[1] * outsz[2])
        cls = cls / float(Num_iter_per_show * props.shape[0] * outsz[0] * outsz[1] * outsz[2])
        
        err_list.append( err )
        cls_list.append( cls )
        it_list.append( i )

        # time
        elapsed = time.time() - start
        print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s, learning rate: %.4f"\
                %(i, err, cls, elapsed, eta*float(outsz[0] * outsz[1] * outsz[2]) )
        # real time visualization
        plt.subplot(331),   plt.imshow(vol_ins[0,0,:,:],       interpolation='nearest', cmap='gray')
        plt.xlabel('input')
        plt.subplot(332),   plt.imshow(props[1,0,:,:],    interpolation='nearest', cmap='gray')
        plt.xlabel('inference')
        plt.subplot(333),   plt.imshow(lbl_outs[1,0,:,:], interpolation='nearest', cmap='gray')
        plt.xlabel('lable')
        plt.subplot(334),   plt.imshow(grdts[1,0,:,:],     interpolation='nearest', cmap='gray')
        plt.xlabel('gradient')
        if is_rebalance:
            plt.subplot(335),   plt.imshow(   rb_weights[1,0,:,:],interpolation='nearest', cmap='gray')
            plt.xlabel('rebalance weight')
        if is_malis:
            plt.subplot(335),   plt.imshow(np.log(malis_weights[1,0,:,:]),interpolation='nearest', cmap='gray')
            plt.xlabel('malis weight (log)')
            plt.subplot(336),   plt.imshow( np.abs(grdts_bm[1,0,:,:] ),interpolation='nearest', cmap='gray')
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
