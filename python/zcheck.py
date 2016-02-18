#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2016
"""
import os
import numpy as np
import utils
import emirt

def check_gradient(pars, net, smp, h=0.001):
    """
    gradient check method:
    http://cs231n.github.io/neural-networks-3/
    """
    # get random sub volume from sample
    vol_ins, lbl_outs, msks, wmsks = smp.get_random_sample()

    # numerical gradient
    # apply the transformations in memory rather than array view
    vol_ins = utils.make_continuous(vol_ins)
    props = net.forward( vol_ins )
    props, cerr, grdts = pars['cost_fn']( props, lbl_outs, msks )

    # shift the input to compute the analytical gradient
    vol_ins1 = dict()
    vol_ins2 = dict()
    for key, val in vol_ins.iteritems():
        vol_ins1[key] = val - h
        vol_ins2[key] = val + h

    props1 = net.forward( vol_ins1 )
    props2 = net.forward( vol_ins2 )
    # compute the analytical gradient
    for key, g in grdts.iteritems():
        lbl = lbl_outs[key]
        prop = props[key]
        prop1 = props1[key]
        prop2 = props2[key]
        ag = (prop2 - prop1)/ (2 * h)
        error = g-ag
        # check the error range
        print "gradient error: ", error

        com = emirt.show.CompareVol((prop[0,...], prop1[0,...], prop2[0,...], g[0,...], ag[0,...]))
        com.vol_compare_slice()

        # check the relative error
        rle = np.abs(ag-g) / (np.maximum(np.abs(ag),np.abs(g)))
        print "relative gradient error: ", rle
        assert error.max < 10*h*h
        assert rle.max() < 0.01

def check_dict_all_zero( d ):
    for v in d.values():
        if np.all(v==0):
            print "all zero!"
            return True
    return False

def check_patch(pars, smp):
    # get random sub volume from sample
    vol_ins, lbl_outs, msks, wmsks = smp.get_random_sample()

    # check the image with ground truth
    fdir = os.path.dirname(pars['train_net'])
    fname = fdir + "/gtruth/net_1.h5"
    if os.path.exists(fname):
        import h5py
        f = h5py.File(fname)
        print "find and check using "+ fname
        stdpre = "/processing/znn/train/patch/"
        for key,val in vol_ins.iteritems():
            ref = f[stdpre+"inputs/"+key]
            assert np.all(val==ref)
        for key,val in lbl_outs.iteritems():
            ref = f[stdpre+"lbls/"+key]
            assert np.all(val==ref)
        f.close()
        print "congrates! patch checking passed!"
    else:
        print "no checking reference file: ", fname


if __name__ == '__main__':
    """
    usage
    ------
    python train.py path/to/config.cfg
    """
    print "main function was removed!"