#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import utils
import cost_fn
import numpy as np
from core import pyznn
import time

def _single_test(net, pars, sample, vn):
    # return errors as a dictionary
    derr = dict()
    vol_ins, lbl_outs, msks, wmsks = sample.get_random_sample()

    # forward pass
    vol_ins = utils.make_continuous(vol_ins)
    props = net.forward( vol_ins )

    # cost function and accumulate errors
    props, derr['err'], grdts = pars['cost_fn']( props, lbl_outs )
    # pixel classification error
    derr['cls'] = cost_fn.get_cls(props, lbl_outs)
    # rand error
    #derr['re'] = pyznn.get_rand_error(props.values()[0], lbl_outs.values()[0])

    if pars['is_malis']:
        malis_weights, rand_errors, num_non_bdr = cost_fn.malis_weight( pars, props, lbl_outs )
        # dictionary of malis classification error
        dmc, dme = utils.get_malis_cost( props, lbl_outs, malis_weights )
        derr['mc'] = dmc.values()[0]
        derr['me'] = dme.values()[0]
    # normalization
    derr['err'] /= vn
    derr['cls'] /= vn
    return props, derr

def znn_test(net, pars, samples, vn, it, lc):
    """
    test the net

    Parameters
    ----------
    net : network
    pars : dict, parameters
    sample : a input and output sample
    vn : number of output voxels
    it : current iteration number
    lc : learning curve

    Returns
    -------
    lc : updated learning curve
    """
    derr = dict()
    derr['it'] = it

    net.set_phase(1)
    test_num = pars['test_num']
    for i in xrange( test_num ):
        props, derr1 = _single_test(net, pars, samples, vn)
        for key, value in derr1.items():
            if derr.has_key(key):
                derr[key] += value
            else:
                derr[key] = value

    net.set_phase(0)
    # normalize
    for key, value in derr.items():
        derr[key] = value / test_num
    # update the learning curve
    lc.append_test( derr )

    if pars['is_malis']:
        print "test iter: %d,     cost: %.3f, pixel error: %.3f, malis cost: %.3f, malis error: %.3f"\
                %(derr['it'], derr['err'], derr['cls'],  derr['me'], derr['mc'])
    else:
        print "test iter: %d,     cost: %.3f, pixel error: %.3f" %(derr['it'], derr['err'], derr['cls'])
    return lc

def run_test(net, pars, smp_tst, vn, i, lc, start, total_time):
    # test the net
    if i%pars['Num_iter_per_test']==0:
        # time accumulation should skip the test
        total_time += time.time() - start
        lc = znn_test(net, pars, smp_tst, vn, i, lc)
        start = time.time()
    return lc, start, total_time