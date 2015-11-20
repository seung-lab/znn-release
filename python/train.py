#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import time
import front_end
import netio
import cost_fn
import test
import utils
import zstatistics
import os
import numpy as np

def main( conf_file='config.cfg', logfile=None ):
    #%% parameters
    print "reading config parameters..."
    config, pars = front_end.parser( conf_file )

    if pars.has_key('logging') and pars['logging']:
        print "recording configuration file..."
        front_end.record_config_file( pars )

        logfile = front_end.make_logfile_name( pars )

    #%% create and initialize the network
    if pars['train_load_net'] and os.path.exists(pars['train_load_net']):
        print "loading network..."
        net = netio.load_network( pars )
        # load existing learning curve
        lc = zstatistics.CLearnCurve( pars['train_load_net'] )
        # the last iteration we want to continue training
        iter_last = lc.get_last_it()
    else:
        if pars['train_seed_net'] and os.path.exists(pars['train_seed_net']):
            print "seeding network..."
            net = netio.load_network( pars, is_seed=True )
        else:
            print "initializing network..."
            net = netio.init_network( pars )
        # initalize a learning curve
        lc = zstatistics.CLearnCurve()
        iter_last = lc.get_last_it()

    # show field of view
    print "field of view: ", net.get_fov()

    # total voxel number of output volumes
    vn = utils.get_total_num(net.get_outputs_setsz())

    # set some parameters
    print 'setting up the network...'
    eta = pars['eta']
    net.set_eta( pars['eta'] )
    net.set_momentum( pars['momentum'] )
    net.set_weight_decay( pars['weight_decay'] )

    # initialize samples
    outsz = pars['train_outsz']
    print "\n\ncreate train samples..."
    smp_trn = front_end.CSamples(config, pars, pars['train_range'], net, outsz, logfile)
    print "\n\ncreate test samples..."
    smp_tst = front_end.CSamples(config, pars, pars['test_range'],  net, outsz, logfile)

    # initialization
    elapsed = 0
    err = 0.0 # cost energy
    cls = 0.0 # pixel classification error
    re = 0.0  # rand error
    if pars['is_malis']:
        malis_cls = 0.0
        malis_eng = 0.0

    # interactive visualization
    if pars['is_visual']:
        import matplotlib.pylab as plt
        plt.ion()
        plt.show()

    print "start training..."
    start = time.time()
    total_time = 0.0
    print "start from ", iter_last+1
    for i in xrange(iter_last+1, pars['Max_iter']+1):
        # get random sub volume from sample
        vol_ins, lbl_outs, msks = smp_trn.get_random_sample()

        # forward pass
        # apply the transformations in memory rather than array view
        vol_ins = utils.make_continuous(vol_ins, dtype=pars['dtype'])
        props = net.forward( vol_ins )

        # cost function and accumulate errors
        props, cerr, grdts = pars['cost_fn']( props, lbl_outs )
        err += cerr
        cls += cost_fn.get_cls(props, lbl_outs)

        # mask process the gradient
        grdts = utils.dict_mul(grdts, msks)

        # run backward pass
        grdts = utils.make_continuous(grdts, dtype=pars['dtype'])
        net.backward( grdts )

        if pars['is_malis'] :
            malis_weights, rand_errors, num_non_bdr = cost_fn.malis_weight(pars, props, lbl_outs)

            print "malis weights: ",  malis_weights
            print "malis weight sum: ", np.sum( malis_weights.values()[0] )
            print "number of non-boundary voxels: ", num_non_bdr

            grdts = utils.dict_mul(grdts, malis_weights)
            # accumulate the rand error
            re += rand_errors.values()[0]
            dmc, dme = utils.get_malis_cost( props, lbl_outs, malis_weights )
            malis_cls += dmc.values()[0]
            malis_eng += dme.values()[0]


        total_time += time.time() - start
        start = time.time()

        # test the net
        if i%pars['Num_iter_per_test']==0:
            lc = test.znn_test(net, pars, smp_tst, vn, i, lc)

        if i%pars['Num_iter_per_show']==0:
            # normalize
            err = err / vn / pars['Num_iter_per_show']
            cls = cls / vn / pars['Num_iter_per_show']
            lc.append_train(i, err, cls)

            # time
            elapsed = total_time / pars['Num_iter_per_show']

            if pars['is_malis']:
                re = re / pars['Num_iter_per_show']
                malis_cls = malis_cls / pars['Num_iter_per_show']
                malis_eng = malis_eng / pars['Num_iter_per_show']
                lc.append_train_rand_error( re )
                lc.append_train_malis_cls( malis_cls )
                lc.append_train_malis_eng( malis_eng )

                show_string = "iteration %d,    err: %.3f, cls: %.3f, re: %.6f, me: %.3f, mc: %.3f, elapsed: %.1f s/iter, learning rate: %.6f"\
                              %(i, err, cls, re, malis_eng, malis_cls, elapsed, eta )
            else:
                show_string = "iteration %d,    err: %.3f, cls: %.3f, elapsed: %.1f s/iter, learning rate: %.6f"\
                    %(i, err, cls, elapsed, eta )

            if pars.has_key('logging') and pars['logging']:
                utils.write_to_log(logfile, show_string)
            print show_string

            if pars['is_visual']:
                # show results To-do: run in a separate thread
                front_end.inter_show(start, lc, eta, vol_ins, props, lbl_outs, grdts, pars)
                if pars['is_rebalance'] and 'aff' not in pars['out_type']:
                    plt.subplot(247)
                    plt.imshow(msks.values()[0][0,0,:,:], \
                               interpolation='nearest', cmap='gray')
                    plt.xlabel('rebalance weight')
                if pars['is_malis']:
                    plt.subplot(248)
                    plt.imshow(malis_weights.values()[0][0,0,:,:], \
                               interpolation='nearest', cmap='gray')
                    plt.xlabel('malis weight (log)')
                plt.pause(2)
                plt.show()
            # reset err and cls
            err = 0
            cls = 0
            re = 0
            if pars['is_malis']:
                malis_cls = 0

            # reset time
            total_time  = 0
            start = time.time()

        if i%pars['Num_iter_per_annealing']==0:
            # anneal factor
            eta = eta * pars['anneal_factor']
            net.set_eta(eta)

        if i%pars['Num_iter_per_save']==0:
            # save network
            netio.save_network(net, pars['train_save_net'], num_iters=i)
            lc.save( pars, elapsed )
            if pars['is_malis']:
                utils.save_malis(malis_weights,  pars['train_save_net'], num_iters=i)

if __name__ == '__main__':
    """
    usage
    ------
    python train.py path/to/config.cfg
    """
    import sys
    if len(sys.argv)>1:
        main( sys.argv[1] )
    else:
        main()
