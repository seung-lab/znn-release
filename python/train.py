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
import utils

def main( conf_file='config.cfg' ):
    #%% parameters
    print "reading data..."
    config, pars = front_end.parser( conf_file )

    #%% create and initialize the network
    outsz = pars['train_outsz']
    print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])

    if pars['train_load_net']:
        print "loading network..."
        net = netio.load_network( pars['train_load_net'], pars['fnet_spec'], outsz, pars['num_threads'])
    else:
        print "initializing network..."
        net = pyznn.CNet(pars['fnet_spec'], outsz, pars['num_threads'], pars['is_optimize'], 0)
    # number of output voxels
    vn = utils.get_total_num(net.get_outputs())
    eta = pars['eta'] / vn
    net.set_eta( eta )
    net.set_momentum( pars['momentum'] )

    # get input and output information
    info_in  = net.get_inputs()
    info_out = net.get_outputs()

    # initialize samples
    print "create input output samples"
    smp_trn = front_end.CSamples(config, pars, pars['train_range'], info_in, info_out)
    smp_tst = front_end.CSamples(config, pars, pars['test_range'],  info_in, info_out)

    eta = pars['eta']
    net.set_eta( eta )
    net.set_momentum( pars['momentum'] )

    # number for normalization
    tn = pars['Num_iter_per_show'] * outsz[0] * outsz[1] * outsz[2]

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
    malis_weights=[]

    # interactive visualization
    plt.ion()
    plt.show()

    start = time.time()
    for i in xrange(1, pars['Max_iter'] ):
        vol_ins, lbl_outs = smp_trn.get_random_sample()

        # forward pass
        vol_ins = utils.make_continuous(vol_ins, dtype='float32')
        props = net.forward( vol_ins )

        # cost function and accumulate errors
        props, cerr, grdts = pars['cost_fn']( props, lbl_outs )
        err = err + cerr
        cls = cls + cost_fn.get_cls(props, lbl_outs)

        # run backward pass
        grdts = utils.make_continuous(grdts, dtype='float32')
        net.backward( grdts )

        if i%pars['Num_iter_per_test']==0:
            # test the net
            terr_list, tcls_list = test.znn_test(net, pars, smp_tst,\
                                            vn, terr_list, tcls_list)
            titr_list.append(i)

        if i%pars['Num_iter_per_show']==0:
            # anneal factor
            eta = eta * pars['anneal_factor']
            net.set_eta(eta)
            # normalize
            err = err / vn / pars['Num_iter_per_show']
            cls = cls / vn / pars['Num_iter_per_show']
            err_list.append( err )
            cls_list.append( cls )
            it_list.append( i )

            if pars['is_visual']:
                # show results To-do: run in a separate thread
                front_end.inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                                        titr_list, terr_list, tcls_list, \
                                        eta, vol_ins, props, lbl_outs, grdts, pars)

            err = err / tn
            cls = cls / tn

            err_list.append( err )
            cls_list.append( cls )
            it_list.append( i )

            # show results To-do: run in a separate thread
            front_end.inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                                    titr_list, terr_list, tcls_list, \
                                    eta, \
                                    vol_ins[0], props[0], lbl_outs[0], grdts[0],pars)
            if pars['is_malis']:
                plt.subplot(248)
                plt.imshow(np.log(malis_weights[0][0,:,:]), interpolation='nearest', cmap='gray')
                plt.xlabel('malis weight (log)')
            # reset err and cls
            err = 0
            cls = 0
            # reset time
            start = time.time()

        if i%pars['Num_iter_per_save']==0:
            # save network
            print "save network"
            netio.save_network(net, pars['train_save_net'], num_iters=i)
            utils.save_statistics( pars, it_list, err_list, cls_list,\
                                    titr_list, terr_list, tcls_list)

if __name__ == '__main__':
    import sys
    if len(sys.argv)>1:
        main( sys.argv[1] )
    else:
        main()
