#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
from front_end import *
import cost_fn
import utils
import os
import emirt
import time
import numpy as np

def main( conf_file='../testsuit/sample/config.cfg', logfile=None ):
    #%% parameters
    print "reading config parameters..."
    config, pars = zconfig.parser( conf_file )

    # random seed
    if pars['is_debug']:
        # use fixed index
        np.random.seed(1)

    if pars.has_key('logging') and pars['logging']:
        print "recording configuration file..."
        zlog.record_config_file( pars )

        logfile = zlog.make_logfile_name( pars )
        print "log file name: ", logfile

    #%% create and initialize the network
    iter_last = 0
    if pars['train_load_net'] and os.path.exists(pars['train_load_net']):
        print "loading network..."
        net = znetio.load_network( pars )
    else:
        print "initializing network..."
        net = znetio.init_network( pars )

    lc = None
    # show field of view
    fov = np.asarray(net.get_fov(), dtype='uint32')

    print "field of view: ", fov

    # total voxel number of output volumes
    vn = utils.get_total_num(net.get_outputs_setsz())

    # set some parameters
    print 'setting up the network...'
    eta = pars['eta']
    net.set_eta( pars['eta'] )
    net.set_momentum( pars['momentum'] )
    net.set_weight_decay( pars['weight_decay'] )

    # initialize samples
    outsz = pars['train_outsz']
    print "\n\ncreate train samples..."
    smp_trn = zsample.CSamples(config, pars, pars['train_range'], net, outsz, logfile)
    print "\n\ncreate test samples..."
    smp_tst = zsample.CSamples(config, pars, pars['test_range'],  net, outsz, logfile)

    # initialization
    elapsed = 0
    err = 0.0 # cost energy
    cls = 0.0 # pixel classification error
    re = 0.0  # rand error
    # number of voxels which accumulate error
    # (if a mask exists)
    num_mask_voxels = 0

    if pars['is_malis']:
        malis_cls = 0.0

    print "start training..."
    start = time.time()
    total_time = 0.0
    print "start from ", iter_last+1

    for i in xrange(iter_last+1, pars['Max_iter']+1):
        # iteration id
        print "iteration: ", i

        # get random sub volume from sample
        vol_ins, lbl_outs, msks, wmsks = smp_trn.get_random_sample()

        # check the patch
        if pars['is_debug']:
            check_patch(pars, fov, i, vol_ins, lbl_outs, \
                        msks, wmsks, is_save=True)

            if check_dict_all_zero( lbl_outs ):
                # forward pass
                # apply the transformations in memory rather than array view
                vol_ins = utils.make_continuous(vol_ins, dtype=pars['dtype'])
                props = net.forward( vol_ins )
                props, cerr, grdts = pars['cost_fn']( props, lbl_outs, msks )
                malis_weights, rand_errors, num_non_bdr = cost_fn.malis_weight(pars, props, lbl_outs)
                utils.inter_save(pars, net, lc, vol_ins, props, lbl_outs, \
                                 grdts, malis_weights, wmsks, elapsed, i)
                raise NameError("all zero groundtruth!")


def check_dict_all_zero( d ):
    for v in d.values():
        if np.all(v==0):
            print "all zero!"
            return True
    return False

def check_patch(pars, fov, i, vol_ins, lbl_outs, \
                msks, wmsks, is_save=False):
    # margin of low and high
    mlow = (fov-1)/2
    mhigh = fov/2

    # get the input and output image
    inimg = vol_ins.values()[0][0,0,:,:]
    if "bound" in pars['out_type']:
        oulbl = lbl_outs.values()[0][0,0,:,:]
        wmsk  = wmsks.values()[0][0,0,:,:]
    else:
        oulbl = lbl_outs.values()[0][2,0,:,:]
        wmsk  = wmsks.values()[0][2,0,:,:]

    # combine them to a RGB image
    # rgb = np.tile(inimg, (3,1,1))
    rgb = np.zeros((3,)+oulbl.shape, dtype='uint8')
    # transform to 0-255
    inimg -= inimg.min()
    inimg = (inimg / inimg.max()) * 255
    inimg = 255 - inimg
    inimg = inimg.astype( 'uint8')
    print "maregin low: ", mlow
    print "margin high: ", mhigh
    inimg = inimg[mlow[1]:-47, mlow[2]:-47]

    oulbl = ((1-oulbl)*255).astype('uint8')
    #rgb[0,:,:] = inimg[margin_low[1]:-margin_high[1], margin_low[2]:-margin_high[2]]
    rgb[0,:,:] = inimg
    rgb[1,:,:] = oulbl

    # rebalance weight
    print "rebalance weight: ", wmsk
    wmsk -= wmsk.min()
    wmsk = (wmsk / wmsk.max()) *255
    wmsk = wmsk.astype('uint8')
    # save the images
    import emirt
    if is_save:
        if 'aff' in pars['out_type']:
            fdir = "../testsuit/affinity/"
        else:
            fdir = "../testsuit/boundary/"
        emirt.emio.imsave(rgb, fdir + "iter_{}_rgb.tif".format(i))
        emirt.emio.imsave(inimg, fdir + "iter_{}_raw.tif".format(i))
        emirt.emio.imsave(wmsk, fdir + "iter_{}_msk.tif".format(i))

    # check the image with ground truth
    fname = fdir + "gtruth/iter_{}_rgb.tif".format(i)
    import os
    if os.path.exists(fname):
        print "find and check using "+ fname
        trgb = emirt.emio.imread( fname )
        assert np.all(trgb == rgb)


if __name__ == '__main__':
    """
    usage
    ------
    python train.py path/to/config.cfg
    """
    import sys
    if len(sys.argv)>1:
        main( sys.argv[1] )
    else:
        main()
