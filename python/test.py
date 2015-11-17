#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import utils
import cost_fn
import numpy as np

def _single_test(net, pars, sample):
    vol_ins, lbl_outs, msks = sample.get_random_sample()

    # forward pass
    vol_ins = utils.make_continuous(vol_ins, dtype=pars['dtype'])
    props = net.forward( vol_ins )

    # cost function and accumulate errors
    props, err, grdts = pars['cost_fn']( props, lbl_outs )
    cls = cost_fn.get_cls(props, lbl_outs)

    re = np.nan
    if pars['is_malis']:
        malis_weights, rand_errors = cost_fn.malis_weight( props, lbl_outs )
        re = rand_errors.values()[0]

    return props, err, cls, re

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
    net.set_phase(1)
    test_num = pars['test_num']
    for i in xrange( test_num ):
        props, cerr, ccls, cre = _single_test(net, pars, samples)
        err += cerr
        cls += ccls
        re  += cre
    net.set_phase(0)
    # normalize
    err = err / vn / test_num
    cls = cls / vn / test_num
    # rand error only need to be normalized by testing time
    re  = re  / test_num
    # update the learning curve
    lc.append_test( it, err, cls )
    lc.append_test_rand_error( re )
    if pars['is_malis']:
        print "test iter: %d,     err: %.3f, cls: %.3f, re: %.6f" \
                %(it, err, cls, re)
    else:
        print "test iter: %d,     err: %.3f, cls: %.3f" \
                %(it, err, cls)
    return lc
