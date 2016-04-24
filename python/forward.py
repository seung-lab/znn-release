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
import numpy as np
import os
from front_end import *
import utils
from emirt import emio

def parse_args( args ):
    config, params = zconfig.parser( args['config'] )

    if args['net']:
        # overwrite the network in config file
        params['forward_net'] = args['net']

    if args['range']:
        params['forward_range'] = utils.parseIntSet( args['range'] )

    return config, params

def batch_forward_pass( config, params, net, verbose=True, sample_ids=None ):
    '''
    Performs a full forward pass for all samples specified within
    a configuration file

    sample_ids should be a list of ints describing the samples to run
    '''

    output_patch_shape = params['forward_outsz']
    sample_outputs = {}
    #Loop over sample range
    for sample in params['forward_range']:
        print "Sample: %d" % sample
        # read image stacks
        # Note: preprocessing included within CSamples
        # See CONSTANTS section above for optionname values
        Dataset = zsample.CSample(config, params, sample, net, \
                                  outsz = output_patch_shape, is_forward=True )
        sample_outputs[sample] = forward_pass( params, Dataset, net )
    return sample_outputs

def forward_pass( params, Dataset, network, verbose=True ):
    '''
    Performs a full forward pass for a given ConfigSample object (Dataset) and
    a given network object.
    '''
    # Making sure loaded images expect same size output volume
    output_vol_shapes = Dataset.output_volume_shape()
    assert output_volume_shape_consistent(output_vol_shapes)
    output_vol_shape = output_vol_shapes.values()[0]
    Output = zsample.ConfigSampleOutput( params, network, output_vol_shape)
    input_num_patches = Dataset.num_patches()
    output_num_patches = Output.num_patches()
    assert num_patches_consistent(input_num_patches, output_num_patches)
    num_patches = output_num_patches.values()[0]

    for i in xrange( num_patches ):
        if verbose:
	    print "Output patch #{} of {}".format(i+1, num_patches) # i is just an index
        input_patches, junk = Dataset.get_next_patch()
	vol_ins = utils.make_continuous(input_patches)
	output = network.forward( vol_ins )
        Output.set_next_patch( output )
        if params['is_check']:
            break
    # softmax if using softmax_loss
    if 'softmax' in params['cost_fn_str']:
        print "softmax filter..."
        Output = run_softmax( Output )

    return Output

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

    for sample_num, output in sample_outputs.iteritems():
        for dataset_name, dataset in output.output_volumes.iteritems():
            num_volumes = dataset.data.shape[0]

            #Consolidated 4d volume
            # hdf5 output for watershed
            h5name = "{}_sample{}_{}.h5".format(prefix, sample_num,	dataset_name)
            print "save output to ", h5name
            import os
            if os.path.exists( h5name ):
                os.remove( h5name )
            emio.imsave(dataset.data, h5name)

            #Constitutent 3d volumes
            # tif file for easy visualization
            for i in range( num_volumes ):
                emio.imsave(dataset.data[i,:,:,:],\
                    "{}_sample{}_{}_{}.tif".format(prefix, sample_num, dataset_name, i))

def main( args ):
    '''
    Script functionality - runs config_forward_pass and saves the
    output volumes
    '''
    config, params = parse_args( args )

    # load network
    net = znetio.load_network( params, train=False, hdf5_filename=params['forward_net'] )

    sample_outputs = batch_forward_pass( config, params, net, verbose=True, sample_ids=params['forward_range'])

    save_sample_outputs( sample_outputs, params['output_prefix'] )

    return sample_outputs

if __name__ == '__main__':
    """
    usage
    ----
    python forward.py -c path/of/config.cfg -r forward_range
    forward_range: the sample ids, such as 1-3,5
    """
    import argparse
    parser = argparse.ArgumentParser(description="ZNN forward pass.")
    parser.add_argument("-c", "--config", required=True, \
                        help="path of configuration file")
    parser.add_argument("-n", "--net", \
                        help="network path")
    parser.add_argument("-r", "--range", help="sample id range, et.al 1-3,5")

    # make the dictionary of arguments
    args = vars( parser.parse_args() )

    if not os.path.exists( args['config'] ):
        raise NameError("config file not exist!")
    if args['net'] is not None:
        if not os.path.exists( args['net'] ):
            raise NameError( "net file do not exist!")

    main( args )
