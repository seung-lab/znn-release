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
from multiprocessing import Process, Queue

def parse_args(args):
    # parse args
    #%% parameters
    if not os.path.exists( args['config'] ):
        raise NameError("config file not exist!")
    else:
        print "reading config parameters..."
        config, pars = zconfig.parser( args["config"] )

    # overwrite the config file parameters from command line
    if args["is_check"]:
        if "yes" == args["is_check"]:
            pars["is_check"] = True
        elif "no" == args["is_check"]:
            pars["is_check"] = False
        else:
            raise NameError("invalid checking option in command line")
    # data type
    if args["dtype"]:
        if "single" in args["dtype"] or "float32" in args["dtype"]:
            pars["dtype"] = "float32"
        elif "double" in args["dtype"] or "float64" in args["dtype"]:
            pars["dtype"] = "float64"
        else:
            raise NameError("invalid data type defined in command line.")

    # random seed
    if pars['is_debug'] or pars['is_check']:
        # use fixed index
        np.random.seed(1)

    if pars.has_key('logging') and pars['logging']:
        print "recording configuration file..."
        zlog.record_config_file( pars )
        logfile = zlog.make_logfile_name( pars )
    else:
        logfile = None

    # check the seed file
    if args['seed']:
        if not os.path.exists(args['seed']):
            import warnings
            warnings.warn("seed file not found! use train_net_prefix of configuration instead.")
        else:
            pars['seed'] = args['seed']
    else:
        pars['seed'] = None
    return config, pars, logfile

# operations after backward pass
def post_backward(history, i, pars, lc, net, vol_ins, props, lbl_outs, grdts, wmsks, start, total_time):
    history = zstatistics.process_history(history, i)
    utils.inter_save(pars, lc, net, vol_ins, props, lbl_outs, grdts, wmsks, i)
    total_time += time.time() - start
    start = time.time()
    return history, lc, start, total_time

def main( args ):
    config, pars, logfile = parse_args(args)
    #%% create and initialize the network
    net, lc = znetio.create_net(pars)

    # total voxel number of output volumes
    vn = utils.get_total_num(net.get_outputs_setsz())

    # initialize samples
    print "\n\ncreate train samples..."
    smp_trn = zsample.CSamples(config, pars, pars['train_range'], net, pars['train_outsz'], logfile)
    print "\n\ncreate test samples..."
    smp_tst = zsample.CSamples(config, pars, pars['test_range'],  net, pars['train_outsz'], logfile)

    # create a queue for storing samples
    qtrn_smp = Queue(1)
    ptrn_smp = Process(target=zsample.put_random_sample, args=(smp_trn, qtrn_smp,))
    ptrn_smp.daemon = True
    ptrn_smp.start()

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


    #p_post_backward = Process(target = post_backward, args=(history, i, pars, lc, net, vol_ins, props, lbl_outs, grdts, wmsks, start, total_time))
    #p_post_backward.daemon = True

    # start time cumulation
    print "start training..."
    start = time.time()
    total_time = 0.0
    print "start from ", iter_last+1
    for it in xrange(iter_last+1, pars['Max_iter']+1):
        # get random sub volume from sample
        vol_ins, lbl_outs, msks, wmsks = qtrn_smp.get()
        # wait for dataprovider finishes
        #ptrn_smp.join()

        # forward pass
        props = net.forward( vol_ins )

        # get gradient and record history
        history, grdts = cost_fn.get_grdt(pars, history, props, lbl_outs, msks, wmsks, vn)

        # run backward pass
        net.backward( grdts )

        # post backward pass processing
        #p_post_backward.start()
        history, net, lc, start, total_time = zstatistics.process_history(pars, history, lc, net, it, start, total_time)
        utils.inter_save(pars, lc, net, vol_ins, props, lbl_outs, grdts, wmsks, it)

        lc, start, total_time = test.run_test(net, pars, smp_tst, vn, it, lc, start, total_time)

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
