#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import time
import matplotlib.pylab as plt
import front_end
import netio
import cost_fn
import test
import utils

def main( conf_file='config.cfg' ):
    #%% parameters
    print "reading config parameters..."
    config, pars = front_end.parser( conf_file )

    #%% create and initialize the network
    outsz = pars['train_outsz']
    print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])

    if pars['train_load_net']:
        print "loading network..."
        net = netio.load_network( pars )
    else:
        print "initializing network..."
        net = netio.init_network( pars )
    # number of output voxels
    print 'setting up the network...'
    vn = utils.get_total_num(net.get_outputs_setsz())
    eta = pars['eta'] #/ vn
    net.set_eta( eta )
    net.set_momentum( pars['momentum'] )

    # initialize samples
    print "\n\ncreate train samples..."
    smp_trn = front_end.CSamples(config, pars, pars['train_range'], net, outsz)
    print "\n\ncreate test samples..."
    smp_tst = front_end.CSamples(config, pars, pars['test_range'],  net, outsz)

    

    # initialization
    err = 0;
    cls = 0;
    err_list = list()
    cls_list = list()
    it_list = list()

    terr_list = list()
    tcls_list = list()
    titr_list = list()

    # interactive visualization
    plt.ion()
    plt.show()

    print "start training..."
    start = time.time()
    for i in xrange(1, pars['Max_iter'] ):
        vol_ins, lbl_outs, msks = smp_trn.get_random_sample()

        # forward pass
        vol_ins = utils.make_continuous(vol_ins, dtype=pars['dtype'])
        
        props = net.forward( vol_ins )
        
        # cost function and accumulate errors
        props, cerr, grdts = pars['cost_fn']( props, lbl_outs )
        err = err + cerr
        cls = cls + cost_fn.get_cls(props, lbl_outs)

        # mask process the gradient
        grdts = utils.dict_mul(grdts, msks)
        
        # run backward pass
        grdts = utils.make_continuous(grdts, dtype=pars['dtype'])
        net.backward( grdts )

        if pars['is_malis'] :
            malis_weights = cost_fn.malis_weight(props, lbl_outs)
            grdts = utils.dict_mul(grdts, malis_weights)

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

            # time
            elapsed = time.time() - start
            print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s, learning rate: %.6f"\
                    %(i, err, cls, elapsed, eta )

            if pars['is_visual']:
                # show results To-do: run in a separate thread
                front_end.inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                                        titr_list, terr_list, tcls_list, \
                                        eta, vol_ins, props, lbl_outs, grdts, pars)
            if pars['is_visual'] and pars['is_rebalance'] and 'aff' not in pars['out_type']:
                plt.subplot(247)
                plt.imshow(msks.values()[0][0,0,:,:], interpolation='nearest', cmap='gray')
                plt.xlabel('rebalance weight')
            if pars['is_visual'] and pars['is_malis']:
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
