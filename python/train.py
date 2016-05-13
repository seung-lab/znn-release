#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import time
from front_end import *
import cost_fn
import utils
import zstatistics
import os
import numpy as np
import test


def main( args ):
    dspec, pars, logfile = zconfig.parse_args(args)
    #%% create and initialize the network, also create learning curve
    net, lc = znetio.create_net(pars)

    # total voxel number of output volumes
    vn = utils.get_total_num(net.get_outputs_setsz())

    # initialize samples
    print "\n\ncreate train samples..."
    smp_trn = zsample.CSamples(dspec, pars, pars['train_range'], net, pars['train_outsz'], logfile)
    print "\n\ncreate test samples..."
    smp_tst = zsample.CSamples(dspec, pars, pars['test_range'],  net, pars['train_outsz'], logfile)

    if pars['is_check']:
        import zcheck
        zcheck.check_patch(pars, smp_trn)

    # initialize history recording
    history = zstatistics.init_history(pars, lc)

    # the last iteration we want to continue training
    iter_last = lc.get_last_it()

    #Saving initial/seeded network
    # get file name
    fname, fname_current = znetio.get_net_fname( pars['train_net_prefix'], iter_last, suffix="init" )
    utils.init_save(pars, lc, net, iter_last)

    # start time cumulation
    print "start training..."
    start = time.time()
    total_time = 0.0
    print "start from ", iter_last+1

    # start data provider and monitor the interuption
    #ptrn_smp.start()
    for it in xrange(iter_last+1, pars['Max_iter']+1):
        # get random sub volume from sample
        vol_ins, lbl_outs, msks, wmsks = smp_trn.get_random_sample()

        # forward pass
        props = net.forward( vol_ins )

        #print props
        # get gradient and record history
        props, grdts, history = cost_fn.get_grdt(pars, history, props, lbl_outs, msks, wmsks, vn)
        #print props
        #print lbl_outs

        # run backward pass
        net.backward( grdts )

        # post backward pass processing
        history, net, lc, start, total_time = zstatistics.process_history(pars, history, \
                                                            lc, net, it, start, total_time)
        utils.inter_save(pars, lc, net, vol_ins, props, \
                         lbl_outs, grdts, wmsks, it)

        lc, start, total_time = test.run_test(net, pars, smp_tst, \
                                              vn, it, lc, start, total_time)

        # stop the iteration at checking mode
        if pars['is_check']:
            print "only need one iteration for checking, stop program..."
            break

if __name__ == '__main__':
    """
    usage
    ------
    python train.py -c path/to/config.cfg -s path/to/seed/net.h5 -k yes
    """
    import argparse

    parser = argparse.ArgumentParser(description="ZNN network training.")
    parser.add_argument("-c", "--config", required=True, \
                        help="path of configuration file.")
    parser.add_argument("-s", "--seed", \
                        help="load an existing network as seed.")
    parser.add_argument("-k", "--is_check", default=None,\
                        help = "do patch matching/gradient check or not. options: yes, no, none")
    parser.add_argument("-d", "--dtype", default=None, \
                        help = "data type. options: float32, float64, single, double")

    # make the dictionary of arguments
    args = vars( parser.parse_args() )
    main( args )
