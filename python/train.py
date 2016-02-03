#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import time
from front_end import *
import cost_fn
import test
import utils
import zstatistics
import os

def main( args ):
    #%% parameters
    print "reading config parameters..."
    config, pars = zconfig.parser( args["config"] )

    if pars.has_key('logging') and pars['logging']:
        print "recording configuration file..."
        zlog.record_config_file( pars )
        logfile = zlog.make_logfile_name( pars )
    else:
        logfile = None

    #%% create and initialize the network
    fnet = znetio.find_load_net( pars['train_net'], args['seed'] )
    net = znetio.load_network( pars, fnet )
    # load existing learning curve
    lc = zstatistics.CLearnCurve( fnet )
    # the last iteration we want to continue training
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
    smp_trn = zsample.CSamples(config, pars, pars['train_range'], net, outsz, logfile)
    print "\n\ncreate test samples..."
    smp_tst = zsample.CSamples(config, pars, pars['test_range'],  net, outsz, logfile)

    # initialization
    elapsed = 0
    err = 0.0 # cost energy
    cls = 0.0 # pixel classification error
    re = 0.0  # rand error
    # number of voxels which accumulate error
    # (if a mask exists)
    num_mask_voxels = 0

    if pars['is_malis']:
        malis_cls = 0.0

    print "start training..."
    start = time.time()
    total_time = 0.0
    print "start from ", iter_last+1

    #Saving initialized network
    if iter_last+1 == 1:
        znetio.save_network(net, pars['train_net'], num_iters=0)
        lc.save( pars, 0.0 )

    for i in xrange(iter_last+1, pars['Max_iter']+1):
        # get random sub volume from sample
        vol_ins, lbl_outs, msks, wmsks = smp_trn.get_random_sample()

        # forward pass
        # apply the transformations in memory rather than array view
        vol_ins = utils.make_continuous(vol_ins, dtype=pars['dtype'])
        props = net.forward( vol_ins )

        # cost function and accumulate errors
        props, cerr, grdts = pars['cost_fn']( props, lbl_outs, msks )
        err += cerr
        cls += cost_fn.get_cls(props, lbl_outs)
        num_mask_voxels += utils.sum_over_dict(msks)

        # gradient reweighting
        grdts = utils.dict_mul( grdts, msks  )
        grdts = utils.dict_mul( grdts, wmsks )

        if pars['is_malis'] :
            malis_weights, rand_errors = cost_fn.malis_weight(pars, props, lbl_outs)
            grdts = utils.dict_mul(grdts, malis_weights)
            # accumulate the rand error
            re += rand_errors.values()[0]
            malis_cls_dict = utils.get_malis_cls( props, lbl_outs, malis_weights )
            malis_cls += malis_cls_dict.values()[0]


        total_time += time.time() - start
        start = time.time()

        # test the net
        if i%pars['Num_iter_per_test']==0:
            lc = test.znn_test(net, pars, smp_tst, vn, i, lc)

        if i%pars['Num_iter_per_show']==0:
            # normalize
            if utils.dict_mask_empty(msks):
                err = err / vn / pars['Num_iter_per_show']
                cls = cls / vn / pars['Num_iter_per_show']
            else:
                err = err / num_mask_voxels / pars['Num_iter_per_show']
                cls = cls / num_mask_voxels / pars['Num_iter_per_show']

            lc.append_train(i, err, cls)

            # time
            elapsed = total_time / pars['Num_iter_per_show']

            if pars['is_malis']:
                re = re / pars['Num_iter_per_show']
                lc.append_train_rand_error( re )
                malis_cls = malis_cls / pars['Num_iter_per_show']
                lc.append_train_malis_cls( malis_cls )

                show_string = "iteration %d,    err: %.3f, cls: %.3f, re: %.6f, mc: %.3f, elapsed: %.1f s/iter, learning rate: %.6f"\
                              %(i, err, cls, re, malis_cls, elapsed, eta )
            else:
                show_string = "iteration %d,    err: %.3f, cls: %.3f, elapsed: %.1f s/iter, learning rate: %.6f"\
                    %(i, err, cls, elapsed, eta )

            if pars.has_key('logging') and pars['logging']:
                utils.write_to_log(logfile, show_string)
            print show_string

            # reset err and cls
            err = 0
            cls = 0
            re = 0
            num_mask_voxels = 0

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
            znetio.save_network(net, pars['train_net'], num_iters=i)
            lc.save( pars, elapsed )
            if pars['is_malis']:
                utils.save_malis(malis_weights,  pars['train_net'], num_iters=i)

        # run backward pass
        grdts = utils.make_continuous(grdts, dtype=pars['dtype'])
        net.backward( grdts )


if __name__ == '__main__':
    """
    usage
    ------
    python train.py -c path/to/config.cfg -s path/to/seed/net.h5
    """
    import argparse

    parser = argparse.ArgumentParser(description="ZNN network training.")
    parser.add_argument("-c", "--config", required=True, \
                        help="path of configuration file")
    parser.add_argument("-s", "--seed", \
                        help="load an existing network as seed")

    # make the dictionary of arguments
    args = vars( parser.parse_args() )

    if not os.path.exists( args['config'] ):
        raise NameError("config file not exist!")

    if args['seed'] and (not os.path.exists(args['seed'])):
        import warnings
        warnings.warn("seed file not found! use train_net of configuration instead.")

    main(args)
