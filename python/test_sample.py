#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
from front_end import *
import utils
import os
import emirt
import time
import numpy as np

def main( conf_file='config.cfg', logfile=None ):
    #%% parameters
    print "reading config parameters..."
    config, pars = zconfig.parser( conf_file )

    if pars.has_key('logging') and pars['logging']:
        print "recording configuration file..."
        zlog.record_config_file( pars )

        logfile = zlog.make_logfile_name( pars )

    #%% create and initialize the network
    iter_last = 0
    if pars['train_load_net'] and os.path.exists(pars['train_load_net']):
        print "loading network..."
        net = znetio.load_network( pars )
    else:
        print "initializing network..."
        net = znetio.init_network( pars )

    # show field of view
    fov = np.asarray(net.get_fov(), dtype='uint32')
    margin_low = (fov-1)/2
    margin_high = fov/2

    print "field of view: ", fov
    print "low margin: ", margin_low
    print "high margin: ", margin_high

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

    # save samples raw and label data for examination
    smp_trn.save_dataset()
    smp_tst.save_dataset()

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

        # get the input and output image
        inimg = vol_ins.values()[0][0,0,:,:]
        oulbl = lbl_outs.values()[0][2,0,:,:]

        inimg = np.copy(inimg)
        # combine them to a RGB image
        # rgb = np.tile(inimg, (3,1,1))
        rgb = np.zeros((3,)+oulbl.shape, dtype='uint8')
        # transform to 0-255
        inimg -= inimg.min()
        inimg = (inimg / inimg.max()) * 255
        inimg = 255 - inimg
        inimg = inimg.astype( 'uint8')
        inimg = inimg[47:-47, 47:-47]

        oulbl = ((1-oulbl)*255).astype('uint8')

        #rgb[0,:,:] = inimg[margin_low[1]:-margin_high[1], margin_low[2]:-margin_high[2]]
        rgb[0,:,:] = inimg
        rgb[1,:,:] = oulbl
        # save the images
        emirt.emio.imsave(rgb, "../testsuit/sample/rgb_{}.tif".format(i))
        emirt.emio.imsave(inimg, "../testsuit/sample/rgb_{}_raw.tif".format(i))
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
