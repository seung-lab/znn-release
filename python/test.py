#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import utils
import cost_fn
import numpy as np

def _single_test(net, pars, sample):
    vol_ins, lbl_outs, msks, wmsks = sample.get_random_sample()

    # forward pass
    vol_ins = utils.make_continuous(vol_ins, dtype=pars['dtype'])
    props = net.forward( vol_ins )

    # cost, gradient, classification error
    costs, grdts = pars['cost_fn']( props, lbl_outs )
    cerrs = cost_fn.get_cls( props, lbl_outs )

    # apply masks
    costs = utils.dict_mul( costs, msks )
    cerrs = utils.dict_mul( cerrs, msks )

    # apply rebalancing weights
    ucost = utils.sum_over_dict( costs )
    costs = utils.dict_mul( costs, wmsks )

    # record keeping
    err = utils.sum_over_dict(costs)
    cls = utils.sum_over_dict(cerrs)
    num_mask_voxels = utils.sum_over_dict(msks)

    # MALIS
    re = 0.0
    malis_cls = 0.0
    if pars['is_malis']:
        malis_weights, rand_errors = cost_fn.malis_weight( pars, props, lbl_outs )
        re = rand_errors.values()[0]
        # dictionary of malis classification error
        mcd = utils.get_malis_cls( props, lbl_outs, malis_weights )
        malis_cls = mcd.values()[0]

    return err, cls, re, malis_cls, num_mask_voxels, ucost

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
    err   = 0.0
    cls   = 0.0
    re    = 0.0
    mc    = 0.0 # malis classification error
    nmv   = 0.0 # number of mask voxels
    ucost = 0.0 # unbalanced cost

    net.set_phase(1)

    test_num = pars['test_num']
    for i in xrange( test_num ):
        serr, scls, sre, smc, snmv, sucost = _single_test(net, pars, samples)
        err   += serr
        cls   += scls
        re    += sre
        mc    += smc
        nmv   += snmv
        ucost += sucost

    net.set_phase(0)

    # normalize
    if nmv > 0:
        err   = err   / nmv
        cls   = cls   / nmv
        ucost = ucost / nmv
    else:
        err   = err   / vn / test_num
        cls   = cls   / vn / test_num
        ucost = ucost / vn / test_num

    # rand error only need to be normalized by testing time
    re  = re  / test_num
    mc  = mc  / test_num
    # update the learning curve
    lc.append_test( it, err, cls, ucost )
    lc.append_test_rand_error( re )
    lc.append_test_malis_cls( mc )
    if pars['is_malis']:
        print "test iter: %d,     err: %.3f, cls: %.3f, re: %.6f, mc: %.3f"\
                %(it, err, cls, re, mc)
    else:
        print "test iter: %d,     err: %.3f, cls: %.3f" \
                %(it, err, cls)
    return lc
