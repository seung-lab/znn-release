#!/usr/bin/env python
__doc__ = """

ZNN Forward Scanner

Kisuk Lee <kisuklee@mit.edu>
Jingpeng Wu <jingpeng.wu@gmail.com>, 2016
"""

from front_end import *
import utils

from emirt import emio

#CONSTANTS
# (configuration file option names)
output_prefix_optionname = 'output_prefix'
range_optionname         = 'forward_range'
outsz_optionname         = 'forward_outsz'
offset_optionname        = 'forward_offset'
grid_optionname          = 'forward_grid'

# def save_outputs(sample_outputs, prefix):
#     '''
#     Writes the resulting output volumes to disk according to the output_prefix
#     '''
#     for sample_num, output in sample_outputs.iteritems():
#         for dataset_name, dataset in output.output_volumes.iteritems():
#             num_volumes = dataset.data.shape[0]

#             #Consolidated 4d volume
#             # hdf5 output for watershed
#             h5name = "{}_sample{}_{}.h5".format(prefix, sample_num, dataset_name)
#             import os
#             if os.path.exists( h5name ):
#                 os.remove( h5name )
#             emio.imsave(dataset.data, h5name)

#             #Constitutent 3d volumes
#             # tif file for easy visualization
#             for i in range( num_volumes ):
#                 emio.imsave(dataset.data[i,:,:,:],\
#                     "{}_sample{}_{}_{}.tif".format(prefix, sample_num, dataset_name, i))

def main( config, sample_ids=None ):

    config, params = zconfig.parser(config)

    # network
    net = znetio.load_network( params, train=False )

    # options
    outsz  = params[outsz_optionname]
    offset = params[offset_optionname]
    grid   = params[grid_optionname]

    if sample_ids is None:
        sample_ids = params[range_optionname]

    for sample in sample_ids:

        print "Sample: %d" % sample

        # create sample
        dataset = zsample.CSample( config, params, sample, net, \
                                   outsz=outsz, is_forward=True )

        # forward scan
        outputs = net.forward_scan( dataset.imgs, spec, offset, grid )

        # TODO(lee):
        #   softmax

        print "Saving Output Volume %d..." % sample
        # save_outputs( outputs, params[output_prefix_optionname] )

if __name__ == '__main__':
    """
    usage
    ----
    python forward_scan.py path/of/config.cfg forward_range
    forward_range: the sample ids, such as 1-3,5
    """
    from sys import argv
    if len(argv)==2:
        main( argv[1] )
    elif len(argv) > 2:
        sample_ids = zconfig.parseIntSet(argv[2])
        main( argv[1], sample_ids )
    else:
        main('config.cfg')
