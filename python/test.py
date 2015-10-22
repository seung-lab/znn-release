#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import utils
import cost_fn

def _single_test(net, pars, sample):
    vol_ins, lbl_outs, msks = sample.get_random_sample()

    # forward pass
    vol_ins = utils.make_continuous(vol_ins, dtype=pars['dtype'])
    props = net.forward( vol_ins )

    # cost function and accumulate errors
    props, err, grdts = pars['cost_fn']( props, lbl_outs )
    cls = cost_fn.get_cls(props, lbl_outs)
    return props, err, cls

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
    net.set_phase(1)
    test_num = pars['test_num']
    for i in xrange( test_num ):
        props, cerr, ccls = _single_test(net, pars, samples)
        err = err + cerr
        cls = cls + ccls
    net.set_phase(0)
    # normalize
    err = err / vn / test_num
    cls = cls / vn / test_num
    # update the learning curve
    lc.append_test( it, err, cls )
    print "test iter: %d,     err: %.3f,  cls: %.3f" \
                %(it, err, cls)
    return lc