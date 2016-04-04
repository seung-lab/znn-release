#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2016
"""
import os
import numpy as np
import utils
import emirt

def check_gradient(pars, net, smp, h=0.00001):
    """
    gradient check method:
    http://cs231n.github.io/neural-networks-3/

    Note that this function is currently not working!
    We should get the gradient from the C++ core, which is not implemented yet.
    """
    # get random sub volume from sample
    vol_ins, lbl_outs, msks, wmsks = smp.get_random_sample()

    # numerical gradient
    # apply the transformations in memory rather than array view
    vol_ins = utils.make_continuous(vol_ins)
    # shift the input to compute the analytical gradient
    vol_ins1 = dict()
    vol_ins2 = dict()
    for key, val in vol_ins.iteritems():
        vol_ins1[key] = val - h
        vol_ins2[key] = val + h
        assert np.any(vol_ins1[key]!=vol_ins2[key])
    props = net.forward( vol_ins )
    props1 = net.forward( vol_ins1 )
    props2 = net.forward( vol_ins2 )
    import copy
    props_tmp, cerr, grdts = pars['cost_fn']( copy.deepcopy(props), lbl_outs, msks )

    # compute the analytical gradient
    for key, g in grdts.iteritems():
        lbl = lbl_outs[key]
        prop = props[key]
        prop1 = props1[key]
        prop2 = props2[key]
        ag = (prop2 - prop1)/ (2 * h)
        error = g-ag

        # label value
        print "ground truth label: ", lbl[0,...]
        print "forward output: ", prop[0,...]
        print "forward output - h: ", prop1[0,...]
        print "forward output + h: ", prop2[0,...]
        print "numerical gradient: ", g[0,...]
        print "analytical gradient: ", ag[0,...]
        # check the error range
        print "gradient error: ", error[0,...]

        com = emirt.show.CompareVol((lbl[0,...], prop[0,...], prop1[0,...], prop2[0,...], g[0,...], ag[0,...]))
        com.vol_compare_slice()

        # check the relative error
        rle = np.abs(ag-g) / (np.maximum(np.abs(ag),np.abs(g)))
        print "relative gradient error: ", rle[0,...]
        assert error.max < 10*h*h
        assert rle.max() < 0.01

def check_dict_all_zero( d ):
    for v in d.values():
        if np.all(v==0):
            print "all zero!"
            return True
    return False

def check_patch(pars, smp):
    print "check patch matching..."
    # get random sub volume from sample
    vol_ins, lbl_outs, msks, wmsks = smp.get_random_sample()

    # check the image with ground truth
    fdir = os.path.dirname(pars['train_net_prefix'])
    if "float32" in pars['dtype']:
        fname = fdir + "/gtruth/net_1_single.h5"
    elif "float64" in pars['dtype']:
        fname = fdir + "/gtruth/net_1_double.h5"
    else:
        raise NameError("invalid data type.")
    if os.path.exists(fname):
        import h5py
        f = h5py.File(fname)
        print "find existing patch: "+ fname
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
