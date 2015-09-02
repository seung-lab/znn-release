#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import pyznn
import time
import matplotlib.pylab as plt
import front_end
import netio
import cost_fn
import test

#%% parameters
print "reading data..."
config, pars = front_end.parser( 'config.cfg' )
smp_trn = front_end.CSamples(pars['train_range'], config, pars)
smp_tst = front_end.CSamples(pars['test_range'],  config, pars)

#%% create and initialize the network
print "initializing network..."
outsz = pars['train_outsz']
print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])
net = pyznn.CNet(pars['fnet_spec'], outsz, pars['num_threads'])
eta = pars['eta'] / float(outsz[0] * outsz[1] * outsz[2])
net.set_eta( eta )
net.set_momentum( pars['momentum'] )

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

terr_list = list()
tcls_list = list()
titr_list = list()
# the temporal weights
rb_weights=[]
malis_weights=[]
grdts_bm=[]

# interactive visualization
plt.ion()
plt.show()

start = time.time()
for i in xrange(1, pars['Max_iter'] ):
    vol_in, lbl_out = smp_trn.get_random_sample( insz, outsz )

    # forward pass
    prop = net.forward( np.ascontiguousarray(vol_in, dtype='float32') ).astype('float32')

    # softmax
    if pars['cost_fn_str']=='multinomial_cross_entropy':
        prop = cost_fn.softmax(prop)

    # cost function and accumulate errors
    cerr, grdt = pars['cost_fn']( prop.astype('float32'), lbl_out.astype('float32') )
    err = err + cerr
    # classification error
    cls = cls + np.count_nonzero( (prop>0.5)!= lbl_out )

    # rebalance
    if pars['is_rebalance']:
        rb_weight = cost_fn.rebalance( lbl_out )
        grdt = grdt * rb_weight
    if pars['is_malis'] :
        malis_weight = cost_fn.malis_weight(prop)
        grdt = grdt * malis_weight

    # run backward pass
    net.backward( np.ascontiguousarray(grdt, dtype='float32') )
    
    if i%pars['Num_iter_per_test']==0:
        # test the net
        terr_list, tcls_list = test.znn_test(net, pars, smp_tst,\
                                insz, outsz, terr_list, tcls_list)
        titr_list.append(i)
        

    if i%pars['Num_iter_per_show']==0:
        # anneal factor
        eta = eta * pars['anneal_factor']
        net.set_eta(eta)
        # normalize
        err = err / float(pars['Num_iter_per_show'] * prop.shape[0] * outsz[0] * outsz[1] * outsz[2])
        cls = cls / float(pars['Num_iter_per_show'] * prop.shape[0] * outsz[0] * outsz[1] * outsz[2])

        err_list.append( err )
        cls_list.append( cls )
        it_list.append( i )
        
        # show results To-do: run in a separate thread
        start, err, cls = front_end.inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                                        titr_list, terr_list, tcls_list, \
                                        eta*float(outsz[0] * outsz[1] * outsz[2]), \
                                        vol_in, prop, lbl_out, grdt, pars, \
                                        rb_weights, malis_weights)
    if i%pars['Num_iter_per_save']==0:
        # save network
        print "save network"
        netio.save_network(net, pars['train_net'], num_iters=i)
