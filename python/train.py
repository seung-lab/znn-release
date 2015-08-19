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
import test

#%% parameters
print "reading data..."
gpars, tpars, fpars = front_end.parser( 'config.cfg' )
vol_orgs, lbl_orgs = front_end.read_tifs(tpars['ftrns'], tpars['flbls'], dp_type=gpars['dp_type'])
tvl_orgs, tlb_orgs = front_end.read_tifs(tpars['ftsts'], tpars['ftlbs'], dp_type=gpars['dp_type'])

#%% create and initialize the network
print "initializing network..."
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
terr_list = list()
tcls_list = list()
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
    vol_in, lbl_out = front_end.get_sample( vol_orgs, insz, lbl_orgs, outsz )
    if tpars['is_data_aug']:
        vol_in, lbl_out = front_end.data_aug( vol_in, lbl_out )
        
    # forward pass
    prop = net.forward( np.ascontiguousarray(vol_in) ).astype('float32')

    # softmax
    if tpars['cost_fn_str']=='multinomial_cross_entropy':
        prop = cost_fn.softmax(prop)
    
    # cost function and accumulate errors
    cerr, grdt = tpars['cost_fn']( prop.astype('float32'), lbl_out.astype('float32') )
    ccls = np.count_nonzero( (prop>0.5)!= lbl_out )
    err = err + cerr
    cls = cls + ccls
    
    # rebalance
    if tpars['is_rebalance']:
        rb_weight = cost_fn.rebalance( lbl_out )
        grdt = grdt * rb_weight
    if tpars['is_malis']:
        grdt_bm = np.copy(grdt)
        malis_weight = cost_fn.malis_weight(prop)
        grdt = grdt * malis_weight 
    
    # run backward pass
    net.backward( np.ascontiguousarray(grdt) )
    
    if i%tpars['Num_iter_per_show']==0:
        # anneal factor
        eta = eta * tpars['anneal_factor']
        net.set_eta(eta)
        # normalize
        err = err / float(tpars['Num_iter_per_show'] * prop.shape[0] * outsz[0] * outsz[1] * outsz[2])
        cls = cls / float(tpars['Num_iter_per_show'] * prop.shape[0] * outsz[0] * outsz[1] * outsz[2])
        
        err_list.append( err )
        cls_list.append( cls )
        it_list.append( i )
        # test the net
        terr_list, tcls_list = test.znn_test(net, tpars, gpars, tvl_orgs, tlb_orgs,\
                                insz, outsz, terr_list, tcls_list)
        # show results To-do: run in a separate thread
        start, err, cls = front_end.inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                                        terr_list, tcls_list, \
                                        eta*float(outsz[0] * outsz[1] * outsz[2]), \
                                        vol_in, prop, lbl_out, grdt, tpars, \
                                        rb_weights, malis_weights, grdt_bm)
        
        # save network
        print "save network"
        front_end.save_network(net, gpars['fnet'])
