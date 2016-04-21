#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import utils
import cost_fn
import numpy as np

def _single_test(net, pars, sample):
    imgs, lbls, msks, wmsks = sample.get_random_sample()

    # forward pass
    imgs = utils.make_continuous(imgs, dtype=pars['dtype'])
    props = net.forward( imgs )

    # convert to softmax affinity
    props,lbls,msks,wmsks = cost_fn.softmax_affinity(props,lbls,msks,wmsks)

    # cost, gradient, classification error
    props, costs, grdts = cost_fn.multinomial_cross_entropy( props, lbls )
    cerrs = cost_fn.get_cls( props, lbls )

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

    return err, cls, num_mask_voxels, ucost

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
    nmv   = 0.0 # number of mask voxels
    ucost = 0.0 # unbalanced cost

    net.set_phase(1)

    test_num = pars['test_num']
    for i in xrange( test_num ):
        serr, scls, snmv, sucost = _single_test(net, pars, samples)
        err   += serr
        cls   += scls
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

    # update the learning curve
    lc.append_test( it, err, cls, ucost )

    print "test iter: %d,     err: %.3f, cls: %.3f" \
                %(it, err, cls)
    return lc
