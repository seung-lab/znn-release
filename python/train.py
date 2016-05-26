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
import numpy as np
from core import pyznn
import shutil

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

def main( args ):
    config, pars, logfile = parse_args(args)
    #%% create and initialize the network
    net, lc = znetio.create_net(pars)

    # total voxel number of output volumes
    vn = utils.get_total_num(net.get_outputs_setsz())

    # initialize samples
    outsz = pars['train_outsz']
    print "\n\ncreate train samples..."
    smp_trn = zsample.CSamples(config, pars, pars['train_range'], net, outsz, logfile)
    print "\n\ncreate test samples..."
    smp_tst = zsample.CSamples(config, pars, pars['test_range'],  net, outsz, logfile)

    if pars['is_check']:
        import zcheck
        zcheck.check_patch(pars, smp_trn)
        # gradient check is not working now.
        # zcheck.check_gradient(pars, net, smp_trn)

    # initialization
    eta = pars['eta']
    elapsed = 0
    err = 0.0 # cost energy
    cls = 0.0 # pixel classification error
    re = 0.0  # rand error
    # number of voxels which accumulate error
    # (if a mask exists)
    num_mask_voxels = 0

    if pars['is_malis']:
        malis_cls = 0.0
        malis_eng = 0.0
    else:
        malis_weights = None

    # the last iteration we want to continue training
    iter_last = lc.get_last_it()

    print "start training..."
    start = time.time()
    total_time = 0.0
    print "start from ", iter_last+1

    #Saving initial/seeded network
    # get file name
    fname, fname_current = znetio.get_net_fname( pars['train_net_prefix'], iter_last, suffix="init" )
    znetio.save_network(net, fname, pars['is_stdio'])
    lc.save( pars, fname, elapsed=0.0, suffix="init_iter{}".format(iter_last) )
    # no nan detected
    nonan = True

    for i in xrange(iter_last+1, pars['Max_iter']+1):
        # time cumulation
        total_time += time.time() - start
        start = time.time()

        # get random sub volume from sample
        vol_ins, lbl_outs, msks, wmsks = smp_trn.get_random_sample()

        # forward pass
        # apply the transformations in memory rather than array view
        vol_ins = utils.make_continuous(vol_ins)
        props = net.forward( vol_ins )

        # cost function and accumulate errors
        props, cerr, grdts = pars['cost_fn']( props, lbl_outs, msks )
        err += cerr
        cls += cost_fn.get_cls(props, lbl_outs)
        # compute rand error
        if pars['is_debug']:
            assert not np.all(lbl_outs.values()[0]==0)
        re  += pyznn.get_rand_error( props.values()[0], lbl_outs.values()[0] )
        num_mask_voxels += utils.sum_over_dict(msks)

        # check whether there is a NaN here!
        if pars['is_debug']:
            nonan = nonan and utils.check_dict_nan(vol_ins)
            nonan = nonan and utils.check_dict_nan(lbl_outs)
            nonan = nonan and utils.check_dict_nan(msks)
            nonan = nonan and utils.check_dict_nan(wmsks)
            nonan = nonan and utils.check_dict_nan(props)
            nonan = nonan and utils.check_dict_nan(grdts)
            if  not nonan:
                utils.inter_save(pars, net, lc, vol_ins, props, lbl_outs, \
                             grdts, malis_weights, wmsks, elapsed, i)
                # stop training
                return

        # gradient reweighting
        grdts = utils.dict_mul( grdts, msks  )
        if pars['rebalance_mode']:
            grdts = utils.dict_mul( grdts, wmsks )

        if pars['is_malis'] :
            malis_weights, rand_errors, num_non_bdr = cost_fn.malis_weight(pars, props, lbl_outs)
            if num_non_bdr<=1:
                # skip this iteration
                continue
            grdts = utils.dict_mul(grdts, malis_weights)
            dmc, dme = utils.get_malis_cost( props, lbl_outs, malis_weights )
            malis_cls += dmc.values()[0]
            malis_eng += dme.values()[0]

        # run backward pass
        grdts = utils.make_continuous(grdts)
        net.backward( grdts )

        total_time += time.time() - start
        start = time.time()

        if i%pars['Num_iter_per_show']==0:
            # time
            elapsed = total_time / pars['Num_iter_per_show']

            # normalize
            if utils.dict_mask_empty(msks):
                err = err / vn / pars['Num_iter_per_show']
                cls = cls / vn / pars['Num_iter_per_show']
            else:
                err = err / num_mask_voxels / pars['Num_iter_per_show']
                cls = cls / num_mask_voxels / pars['Num_iter_per_show']
            re = re / pars['Num_iter_per_show']
            lc.append_train(i, err, cls, re)

            if pars['is_malis']:
                malis_cls = malis_cls / pars['Num_iter_per_show']
                malis_eng = malis_eng / pars['Num_iter_per_show']
                lc.append_train_malis_cls( malis_cls )
                lc.append_train_malis_eng( malis_eng )

                show_string = "update %d,    cost: %.3f, pixel error: %.3f, rand error: %.3f, me: %.3f, mc: %.3f, elapsed: %.1f s/iter, learning rate: %.5f"\
                              %(i, err, cls, re, malis_eng, malis_cls, elapsed, eta )
            else:
                show_string = "update %d,    cost: %.3f, pixel error: %.3f, rand error: %.3f, elapsed: %.1f s/iter, learning rate: %.5f"\
                    %(i, err, cls, re, elapsed, eta )

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

        # test the net
        if i%pars['Num_iter_per_test']==0:
            # time accumulation should skip the test
            total_time += time.time() - start
            lc = test.znn_test(net, pars, smp_tst, vn, i, lc)
            start = time.time()

        if i%pars['Num_iter_per_save']==0:
            utils.inter_save(pars, net, lc, vol_ins, props, lbl_outs, \
                             grdts, malis_weights, wmsks, elapsed, i)


        if i%pars['Num_iter_per_annealing']==0:
            # anneal factor
            eta = eta * pars['anneal_factor']
            net.set_eta(eta)

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
