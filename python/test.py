#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import utils
import cost_fn
import numpy as np
from core import pyznn

def _single_test(net, pars, sample):
    vol_ins, lbl_outs, msks, wmsks = sample.get_random_sample()

    # forward pass
    vol_ins = utils.make_continuous(vol_ins)
    props = net.forward( vol_ins )

    # cost function and accumulate errors
    props, err, grdts = pars['cost_fn']( props, lbl_outs )
    # pixel classification error
    cls = cost_fn.get_cls(props, lbl_outs)
    # rand error
    re = pyznn.get_rand_error(props.values()[0], lbl_outs.values()[0])

    malis_cls = 0.0
    malis_eng = 0.0

    if pars['is_malis']:
        malis_weights, rand_errors, num_non_bdr = cost_fn.malis_weight( pars, props, lbl_outs )
        # dictionary of malis classification error
        dmc, dme = utils.get_malis_cost( props, lbl_outs, malis_weights )
        malis_cls = dmc.values()[0]
        malis_eng = dme.values()[0]
    return props, err, cls, re, malis_cls, malis_eng

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
    err = 0.0
    cls = 0.0
    re = 0.0
    # malis classification error
    mc = 0.0
    me = 0.0

    net.set_phase(1)
    test_num = pars['test_num']
    for i in xrange( test_num ):
        props, cerr, ccls, cre, cmc, cme = _single_test(net, pars, samples)
        err += cerr
        cls += ccls
        re  += cre
        mc  += cmc
        me  += cme
    net.set_phase(0)
    # normalize
    err = err / vn / test_num
    cls = cls / vn / test_num
    # rand error only need to be normalized by testing time
    re  = re  / test_num
    mc  = mc  / test_num
    me  = me  / test_num
    # update the learning curve
    lc.append_test( it, err, cls, re )
    lc.append_test_malis_cls( mc )
    lc.append_test_malis_eng( me )

    if pars['is_malis']:
        print "test iter: %d,     cost: %.3f, pixel error: %.3f, rand error: %.6f, malis cost: %.3f, malis error: %.3f"\
                %(it, err, cls, re, me, mc)
    else:
        print "test iter: %d,     cost: %.3f, pixel error: %.3f" %(it, err, cls)
    return lc
