#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import time
import matplotlib.pylab as plt
import front_end
import netio
import cost_fn
import test
import utils
import zstatistics
import os

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
    else:
        if pars['train_seed_net'] and os.path.exists(pars['train_seed_net']):
            print "seeding network..."
            net = netio.seed_network( pars, is_seed=True )
        else:
            print "initializing network..."
            net = netio.init_network( pars )
        # initalize a learning curve
        lc = zstatistics.CLearnCurve()

    # show field of view
    print "field of view: ", net.get_fov()
    print "output volume info: ", net.get_outputs_setsz()

    # set some parameters
    print 'setting up the network...'
    vn = utils.get_total_num(net.get_outputs_setsz())
    eta = pars['eta'] #/ vn
    net.set_eta( eta )
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
    err = 0
    cls = 0

    # interactive visualization
    plt.ion()
    plt.show()

    # the last iteration we want to continue training
    iter_last = lc.get_last_it()


    print "start training..."
    start = time.time()
    print "start from ", iter_last+1
    for i in xrange(iter_last+1, pars['Max_iter']+1):
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
            lc = test.znn_test(net, pars, smp_tst, vn, i, lc)

        if i%pars['Num_iter_per_show']==0:
            # anneal factor
            eta = eta * pars['anneal_factor']
            net.set_eta(eta)
            # normalize
            err = err / vn / pars['Num_iter_per_show']
            cls = cls / vn / pars['Num_iter_per_show']
            lc.append_train(i, err, cls)

            # time
            elapsed = time.time() - start
            elapsed = elapsed / pars['Num_iter_per_show']

            show_string = "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s/iter, learning rate: %.6f"\
                    %(i, err, cls, elapsed, eta )

            if pars.has_key('logging') and pars['logging']:
                utils.write_to_log(logfile, show_string)
            print show_string

            if pars['is_visual']:
                # show results To-do: run in a separate thread
                front_end.inter_show(start, lc, eta, vol_ins, props, lbl_outs, grdts, pars)
                if pars['is_rebalance'] and 'aff' not in pars['out_type']:
                    plt.subplot(247)
                    plt.imshow(msks.values()[0][0,0,:,:], interpolation='nearest', cmap='gray')
                    plt.xlabel('rebalance weight')
                if pars['is_malis']:
                    plt.subplot(248)
                    plt.imshow(malis_weights.values()[0][0,0,:,:], interpolation='nearest', cmap='gray')
                    plt.xlabel('malis weight (log)')
                plt.pause(2)
                plt.show()
            # reset err and cls
            err = 0
            cls = 0
            # reset time
            start = time.time()

        if i%pars['Num_iter_per_save']==0:
            # save network
            netio.save_network(net, pars['train_save_net'], num_iters=i)
            lc.save( pars, elapsed )

if __name__ == '__main__':
    """
    usage:
    ------
    python train.py path/to/config.cfg
    """
    import sys
    if len(sys.argv)>1:
        main( sys.argv[1] )
    else:
        main()
