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

#%% parameters
gpars, tpars, fpars = front_end.parser( 'config.cfg' )
vol_orgs, lbl_orgs = front_end.read_tifs(tpars['ftrns'], tpars['flbls'])

#%% create and initialize the network
outsz = tpars['outsz']
print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])
net = pyznn.CNet(gpars['fnet_spec'], outsz, gpars['num_threads'])
eta = tpars['eta'] / float(outsz[0] * outsz[1] * outsz[2])
net.set_eta( eta )
net.set_momentum( tpars['momentum'] )

#%% compute inputsize and get input
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1

# initialization
err = 0;
cls = 0;
err_list = list()
cls_list = list()
it_list = list()
# the temporal weights
rb_weights=[] 
malis_weights=[]
grdts_bm=[]

# interactive visualization
plt.ion()
plt.show()

start = time.time()
for i in xrange(1, tpars['Max_iter'] ):
    vol_ins, lbl_outs = front_end.get_sample( vol_orgs, insz, lbl_orgs, outsz, dp_type=gpars['dp_type'] )
    if tpars['is_data_aug']:
        vol_ins, lbl_outs = front_end.data_aug( vol_ins, lbl_outs )
        
    # forward pass
    props = net.forward( np.ascontiguousarray(vol_ins) ).astype('float32')

    # softmax
    if tpars['cost_fn_str']=='multinomial_cross_entropy':
        props = cost_fn.softmax(props)
    
    # cost function and accumulate errors
    cerr, grdts = tpars['cost_fn']( props.astype('float32'), lbl_outs.astype('float32') )
    ccls = np.count_nonzero( (props>0.5)!= lbl_outs )
    err = err + cerr
    cls = cls + ccls
    
    # rebalance
    if tpars['is_rebalance']:
        rb_weights = cost_fn.rebalance( lbl_outs )
        grdts = grdts * rb_weights
    if tpars['is_malis']:
        grdts_bm = np.copy(grdts)
        malis_weights = cost_fn.malis_weights(props)
        grdts = grdts * malis_weights 
    
    # run backward pass
    net.backward( np.ascontiguousarray(grdts) )
    
    if i%tpars['Num_iter_per_show']==0:
        # anneal factor
        eta = eta * tpars['anneal_factor']
        net.set_eta(eta)
        # normalize
        err = err / float(tpars['Num_iter_per_show'] * props.shape[0] * outsz[0] * outsz[1] * outsz[2])
        cls = cls / float(tpars['Num_iter_per_show'] * props.shape[0] * outsz[0] * outsz[1] * outsz[2])
        
        err_list.append( err )
        cls_list.append( cls )
        it_list.append( i )
        start, err, cls = front_end.inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                                        eta*float(outsz[0] * outsz[1] * outsz[2]), \
                                        vol_ins, props, lbl_outs, grdts, tpars, \
                                        rb_weights, malis_weights, grdts_bm)
        
        # save network
        print "save network"
        front_end.save_network(net, gpars['fnet'])
