#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
from front_end import *
import utils
import os
import emirt

def main( conf_file='config.cfg', logfile=None ):
    #%% parameters
    print "reading config parameters..."
    config, pars = zconfig.parser( conf_file )

    if pars.has_key('logging') and pars['logging']:
        print "recording configuration file..."
        zlog.record_config_file( pars )

        logfile = zlog.make_logfile_name( pars )

    #%% create and initialize the network
    if pars['train_load_net'] and os.path.exists(pars['train_load_net']):
        print "loading network..."
        net = znetio.load_network( pars )
        # load existing learning curve
        lc = zstatistics.CLearnCurve( pars['train_load_net'] )
        # the last iteration we want to continue training
        iter_last = lc.get_last_it()
    else:
        if pars['train_seed_net'] and os.path.exists(pars['train_seed_net']):
            print "seeding network..."
            net = znetio.load_network( pars, is_seed=True )
        else:
            print "initializing network..."
            net = znetio.init_network( pars )
        # initalize a learning curve
        lc = zstatistics.CLearnCurve()
        iter_last = lc.get_last_it()

    # show field of view
    print "field of view: ", net.get_fov()

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

    #Saving initialized network
    if iter_last+1 == 1:
        znetio.save_network(net, pars['train_save_net'], num_iters=0)

    for i in xrange(iter_last+1, pars['Max_iter']+1):
        # iteration id
        print "iteration: ", i

        # get random sub volume from sample
        vol_ins, lbl_outs, msks, wmsks = smp_trn.get_random_sample()

        # get the input and output image
        inimg = vol_ins.values()[2,0,:,:]
        oulbl = vol_outs.values()[2,0,:,:]

        # combine them to a RGB image
        rgb = np.tile(inimg, (3,1,1), dtype='uint8')
        rgb[0,:,:][oulbl==0] = 128
        # save the images
        emirt.emio.imsave(inimg, "../experiments/testsample/rgb_raw_lbl_{}.tif".format(i))

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
