#!/usr/bin/env python
__doc__ = """

ZNN Forward Scanner

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

import numpy as np

from front_end import *
import utils

from emirt import emio

def save_outputs( outputs, prefix, sample ):
    for name, data in outputs.iteritems():
        parts = name.split(":")
        emio.imsave( data[0,:,:,:], \
            "{}_sample{}_{}_{}.tif".format(prefix, sample, parts[0], parts[1]))

def main( config, sample_ids=None, scan_spec=None ):

    config, params = zconfig.parser(config)

    if sample_ids is None:
        sample_ids = params['forward_range']

    if scan_spec is None:
        scan_spec = params['fnet_spec']

    # network
    net = znetio.load_network( params, train=False )

    # options
    outsz  = params['forward_outsz']
    offset = np.array([0,0,0],dtype=np.int64)
    grid   = np.array([0,0,0],dtype=np.int64)

    if 'forward_offset' in params:
        offset = params['forward_offset']

    if 'forward_grid' in params:
        grid = params['forward_grid']

    for sample in sample_ids:
        print "Sample: %d" % sample

        dataset = zsample.CSample( config, params, sample, net, \
                                   outsz=outsz, is_forward=True )

        outputs = net.forward_scan( dataset.imgs, scan_spec, offset, grid )

        print "Saving Output Volume %d..." % sample
        save_outputs( outputs, params['output_prefix'], sample )

if __name__ == '__main__':
    """
    usage
    ----
    python forward_scan.py path/of/config.cfg forward_range scan_spec

        forward_range:  the sample ids, such as 1-3,5
        scan_spec
    """
    from sys import argv
    if len(argv) == 2:
        main( argv[1] )
    elif len(argv) > 2:
        sample_ids = zconfig.parseIntSet(argv[2])
        if len(argv) > 3:
            main( argv[1], sample_ids, argv[3] )
        else:
            main( argv[1], sample_ids )
    else:
        main('config.cfg')