#!/usr/bin/env python
__doc__ = """
test whether the sample input and output subvolume matches.
modified from train.py
use the same volume as image and label, and compare the output volume directly.
set sample image and label files

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import time
import matplotlib.pylab as plt
import front_end
import netio
import cost_fn
import test
import utils

def main( conf_file='config.cfg' ):
    #%% parameters
    print "reading config parameters..."
    config, pars = front_end.parser( conf_file )

    #%% create and initialize the network
    outsz = pars['train_outsz']
    print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])

    if pars['train_load_net']:
        print "loading network..."
        net = netio.load_network( pars )
    else:
        print "initializing network..."
        net = netio.init_network( pars )
    # number of output voxels
    print 'setting up the network...'
    vn = utils.get_total_num(net.get_outputs_setsz())
    eta = pars['eta'] #/ vn
    net.set_eta( eta )
    net.set_momentum( pars['momentum'] )

    # initialize samples
    print "\n\ncreate train samples..."
    smp_trn = front_end.CSamples(config, pars, pars['train_range'], net, outsz)
    print "\n\ncreate test samples..."
    smp_tst = front_end.CSamples(config, pars, pars['test_range'],  net, outsz)

    print "start training..."
    start = time.time()
    for i in xrange(1, pars['Max_iter'] ):
        print "iteration ", i
        vol_ins, lbl_outs, msks = smp_trn.get_random_sample()

        vi = vol_ins.values()[0][0,:,18:-18,18:-18]
        vo = lbl_outs.values()[0][0,:,:,:]
        print "volume input: ", vi
        print "volume output: ", vo
        print "vi shape: ", vi.shape
        print "vo shape: ", vo.shape
        assert( vi.shape == vo.shape )
        if np.all(vi==vo):
            print "input matches output"


if __name__ == '__main__':
    import sys
    if len(sys.argv)>1:
        main( sys.argv[1] )
    else:
        main()
