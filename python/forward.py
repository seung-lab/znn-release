#!/usr/bin/env python
__doc__ = """

ZNN Full Forward-Pass Computation

 This module computes the propogation of activation through a
 ZNN neural network. Its command-line/script functionality produces the
 network output for the entirety of sample volumes specified within a
 configuration file (under the option 'forward_range'), opposed to
 processing single output patches.

 The resulting arrays are then saved to disk by the output_prefix option.

 For example, the output_prefix 'out' and one data sample would lead to files saved under
 out_sample1_output_0.tif, out_sample1_output_1.tif, etc. for each sample specified within the
 configuration file, and each constituent 3d volume of the .

 The module also features functions for generating the full output volume
 for a given input np array.

Inputs:

	-Configuration File Name

Main Outputs:

	-Saved .tif/h5 files for each sample within the configuration file

Nicholas Turner <nturner@cs.princeton.edu>
Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
#TODO- Better argument handling

import numpy as np

from front_end import *
import utils

from emirt import emio

#CONSTANTS
# (configuration file option names)
output_prefix_optionname = 'output_prefix'
range_optionname = 'forward_range'
outsz_optionname = 'forward_outsz'


def config_forward_pass( config, params, verbose=True, sample_ids=None, aug_ids=None ):
    '''
    Performs a full forward pass for all samples specified within
    a configuration file

    sample_ids should be a list of ints describing the samples to run
    '''
    # set command line sample ids
    if sample_ids is not None:
        params[range_optionname] = sample_ids

    # test time augmentation
    if aug_ids is None:
        aug_ids = [0]

    # load network
    net = znetio.load_network( params, train=False )
    output_patch_shape = params[outsz_optionname]
    sample_outputs = {}
    #Loop over sample range
    for sample in params[range_optionname]:
        print "Sample: %d" % sample

        # read image stacks
        # Note: preprocessing included within CSamples
        # See CONSTANTS section above for optionname values
        Dataset = zsample.CSample(config, params, sample, net, \
                                  outsz = output_patch_shape, is_forward=True )

        for aug_id in aug_ids:
            print "Augmentation: %d" % aug_id

            output = generate_full_output(Dataset, net, params, params['dtype'],
                                          verbose=True, aug_id=aug_id)

            # softmax if using softmax_loss
            if 'softmax' in params['cost_fn_str']:
                output = run_softmax(output)

            # save output
            print "Saving Output Volume %d (augmentation %d)..." % (sample, aug_id)
            save_sample_output(sample, output, params[output_prefix_optionname], aug_id)

    return sample_outputs

def run_softmax( sample_output ):
    '''
    Performs a softmax calculation over the output volumes for a
    given sample output
    '''
    from cost_fn import softmax

    for dname, dataset in sample_output.output_volumes.iteritems():

		props = {'dataset':dataset.data}
		props = softmax(props)
		dataset.data = props.values()[0]
		sample_output.output_volumes[dname] = dataset

    return sample_output

def generate_full_output( Dataset, network, params, dtype='float32',
                          verbose=True, aug_id=0 ):
	'''
	Performs a full forward pass for a given ConfigSample object (Dataset) and
	a given network object.
	'''

	# Making sure loaded images expect same size output volume
	output_vol_shapes = Dataset.output_volume_shape()
	assert output_volume_shape_consistent(output_vol_shapes)
	output_vol_shape = output_vol_shapes.values()[0]

	Output = zsample.ConfigSampleOutput( params, network, output_vol_shape, dtype )

	input_num_patches = Dataset.num_patches()
	output_num_patches = Output.num_patches()

	assert num_patches_consistent(input_num_patches, output_num_patches)

	num_patches = output_num_patches.values()[0]

	for i in xrange( num_patches ):

		if verbose:
			print "Output patch #{} of {}".format(i+1, num_patches) # i is just an index

		input_patches, junk = Dataset.get_next_patch()

        # test time augmentation
        for key, input_patch in input_patches.iteritems():
            # convert integer to binary number
            rft = np.array([int(x) for x in bin(aug_id)[2:].zfill(4)])
            input_patches[key] = utils.data_aug_transform( input_patch, rft )

        vol_ins = utils.make_continuous(input_patches, dtype=dtype)

        output = network.forward( vol_ins )

        Output.set_next_patch( output )

	return Output

def output_volume_shape_consistent( output_vol_shapes ):
	'''
	Returns whether the dictionary of output shapes passed to the function
	contains the same array for each entry

	Here, this encodes whether all of the input volumes agree on the
	size of the output volume (disagreement is a bad sign...)
	'''
	#output_vol_shapes should be a dict
	shapes = output_vol_shapes.values()
	assert len(shapes) > 0

	return all( [np.all(shape == shapes[0]) for shape in shapes] )

def num_patches_consistent( input_patch_count, output_patch_count ):
	'''
	Returns whether the dictionaries of patch counts all agree throughout
	each entry.
	'''

	#These should be dicts as well
	input_counts = input_patch_count.values()
	output_counts = output_patch_count.values()

	assert len(input_counts) > 0 and len(output_counts) > 0

	return all( [count == input_counts[0] for count in input_counts + output_counts])

def save_sample_outputs(sample_outputs, prefix):
    '''
    Writes the resulting output volumes to disk according to the
    output_prefix
    '''
    for sample_num, outputs in sample_outputs.iteritems():
        for aug_id, output in outputs.iteritems():
            save_sample_output(sample_num, output, prefix, aug_id)

def save_sample_output(sample_num, output, prefix, aug_id=None):
    for dataset_name, dataset in output.output_volumes.iteritems():
        num_volumes = dataset.data.shape[0]

        fprefix = "{}_sample{}_{}".format(prefix, sample_num, dataset_name)

        # Add postfix when test time augmentation is on
        aug_postfix = ""
        if (aug_id is not None) and (aug_id>0):
            aug_postfix = "_aug{}".format(aug_id)

        # Consolidated 4d volume
        # hdf5 output for watershed
        h5name = fprefix + aug_postfix + ".h5"
        import os
        if os.path.exists( h5name ):
            os.remove( h5name )
        emio.imsave(dataset.data, h5name)

        # Constitutent 3d volumes
        # tif file for easy visualization
        for i in range( num_volumes ):
            tifname = fprefix + "_{}".format(i) + aug_postfix + ".tif"
            emio.imsave(dataset.data[i,:,:,:], tifname)


def main( config_filename, sample_ids=None, aug_ids=None ):
    '''
    Script functionality - runs config_forward_pass and saves the
    output volumes
    '''
    config, params = zconfig.parser( config_filename )

    if sample_ids is None:
    	sample_ids = params[range_optionname]

    config_forward_pass( config, params, True, sample_ids, aug_ids )

    # for sample_id in sample_ids:

    # 	output_volume = config_forward_pass( config, params, verbose=True, sample_ids=[sample_id] )

    # 	print "Saving Output Volume %d..." % sample_id
    # 	save_sample_outputs( output_volume, params[output_prefix_optionname] )

if __name__ == '__main__':
    """
    usage
    ----
    python forward.py path/of/config.cfg forward_range
    forward_range: the sample ids, such as 1-3,5
    aug_range: test time augmentation range, such as 1-4
    """
    from sys import argv
    if len(argv)==2:
        main( argv[1] )
    elif len(argv) > 2:
        sample_ids = zconfig.parseIntSet(argv[2])
        if len(argv > 3):
            aug_ids = zconfig.parseIntSet(argv[3])
            main( argv[1], sample_ids=sample_ids, aug_ids=aug_ids )
        else:
            main( argv[1], sample_ids=sample_ids )
    else:
        main('config.cfg')
