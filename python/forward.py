#!/usr/bin/env python
__doc__ = """

ZNN Full Forward-Pass Computation

 This module computes the propogation of activation through a 
 ZNN neural network. Its command-line/script functionality produces the
 network output for the entirety of sample volumes specified within a 
 configuration file (under the option 'forward_range'), opposed to 
 processing single output patches. 

 The resulting arrays are then saved to disk by the output_prefix option. 

 For example, the output_prefix 'out' would lead to files saved under
 out_1.tif, out_2.tif, etc. for each sample specified within the
 configuration file.

 The module also features functions for generating the full output volume
 for a given input np array.

Inputs:

	-Configuration File Name
	
Main Outputs:

	-Saved .tif files for each sample within the configuration file

Nicholas Turner <nturner@cs.princeton.edu>
Jingpeng Wu <jingpeng.wu@gmail.com>, 2015

TODO- Better argument handling
"""

import numpy as np

import front_end, netio

from emirt import emio
from utils import loa_as_continue

#CONSTANTS 
# (configuration file option names)
output_prefix_optionname = 'output_prefix'
range_optionname = 'forward_range'
outsz_optionname = 'forward_outsz'
net_optionname = 'forward_net'
specfile_optionname = 'fnet_spec'
threads_optionname = 'num_threads'

def correct_output_patch_shape( output_patch_config_shape, net ):
	'''
	Returns a 4d version of the output shape array. Always replaces
	the 4th dimension with the num_output_vols specified

	I doubt this is necessary anymore, but might be useful soon
	'''

	num_output_vols = net.get_output_num()

	if output_patch_config_shape.size == 4:

		return np.hstack( (num_output_vols, output_patch_config_shape[1:]) )

	elif output_patch_config_shape.size == 3:

		return np.hstack( (num_output_vols, output_patch_config_shape) )

def input_patch_shape( output_patch_shape, fov ):
	'''Determines the size of the input patch to feed into the network'''
	if output_patch_shape.size == 3:

		return output_patch_shape + fov - 1

	else: #len == 4

		return np.hstack( (output_patch_shape[0],
					output_patch_shape[-3:] + fov - 1) )

def output_vol_shape( input_vol_shape, net ):
	'''
	Derives the resulting shape of the full volume returned by the 
	full forward pass
	'''
	fov = np.asarray(net.get_fov())
	num_output_vols = net.get_output_num()

	return np.hstack( (num_output_vols, input_vol_shape[-3:] - fov + 1 ) )

def num_output_patches( output_vol_shape, output_patch_shape ):
	'''
	One way to derive the number of output patches in the resulting volume.
	
	Mostly used for unit testing at this point

	Restricts calculation to 3d shape
	'''

	#3d Shape restriction
	output_vol_shape = output_vol_shape[-3:]
	output_patch_shape = output_patch_shape[-3:]

	# # per axis  = dim length / patch length (rounded up)
	num_per_axis = np.ceil(output_vol_shape / output_patch_shape.astype(np.float64))
	return int(np.prod(num_per_axis))

def patch_bounds( input_vol_width, output_patch_width, fov_width ):
	'''
	Returns the bounds of one axis for a given input volume width and 
	output patch width
	'''

	bounds = []

	beginning = 0
	ending = output_patch_width + fov_width - 1

	while ending < input_vol_width:

		bounds.append(
			( beginning, ending )
			)

		beginning += output_patch_width
		ending += output_patch_width

	#last bound
	bounds.append(
		( input_vol_width - (output_patch_width + fov_width - 1), input_vol_width)
		)

	return bounds

def input_patch_bounds( input_vol_shape, output_patch_shape, fov ):
	'''
	Finds the bounds for each input patch given the input volume shape,
	the network fov, and the output patch shape

	Restricts calculation to 3d shape
	'''

	#3d Shape restriction
	input_vol_shape = input_vol_shape[-3:]
	output_patch_shape = output_patch_shape[-3:]

	#Decomposing into a similar problem for each axis
	z_bounds = patch_bounds(input_vol_shape[0], output_patch_shape[0], fov[0])
	y_bounds = patch_bounds(input_vol_shape[1], output_patch_shape[1], fov[1])
	x_bounds = patch_bounds(input_vol_shape[2], output_patch_shape[2], fov[2])

	#And then recombining the subproblems
	bounds = []
	for z in z_bounds:
		for y in y_bounds:
			for x in x_bounds:
				bounds.append(
						(
						#beginning for each axis
						(z[0],y[0],x[0]),
						#ending for each axis
						(z[1],y[1],x[1])
						)
					)

	return bounds
	
def output_patch_bounds( output_vol_shape, output_patch_shape ):
	'''
	Finds the bounds for each output patch given the output volume shape
	and the output patch shape
	'''
	#Exact same problem as the input bounds, where the 'fov' is equal to 1
	# in all dimensions
	return input_patch_bounds(output_vol_shape, 
				output_patch_shape, 
				# default type for ones is float, which converts indices
				# to floats down the line and causes problems within patch_bounds
				# (tries to index an array with a float)
				np.ones(output_patch_shape.shape, dtype='uint32'))

def generate_output_volume( input_vol, output_patch_shape, net, verbose=True ):
	'''
	Generates a full output volume for a given input volume - the main 
	functionality of the module

	The shape of the output patch may be specified in 3d or 4d, but we'll
	overwrite any 4d specification with the output size of the net anyway
	'''

	#Making the input volume 4d
	if len(input_vol.shape) == 3:
		input_vol = input_vol.reshape( np.hstack((1,input_vol.shape)) )

	#Initialize output volume
	output_vol = np.zeros( 
			output_vol_shape(input_vol.shape, net), 
			dtype=np.float32)

	#Derive bounds of input and output patches
	fov = np.asarray(net.get_fov())
	input_bounds = input_patch_bounds( input_vol.shape, 
						output_patch_shape,
						fov )
	output_bounds = output_patch_bounds( output_vol.shape,
						output_patch_shape )

	#Stupidity checks (so far so good!)
	assert num_output_patches( output_vol.shape, output_patch_shape ) == len(output_bounds)
	assert len( input_bounds ) == len( output_bounds )

	fov = np.asarray(net.get_fov())
	num_patches = len(input_bounds)

	for i in xrange( num_patches ):

		if verbose:
			print "Output patch #{} out of {}:".format(i+1, num_patches) # i is just an index

		input_beginning = 	input_bounds[i][0]
		input_end = 		input_bounds[i][1]
		if verbose:
			print "Input Volume [{}] to [{}]".format(input_beginning, input_end)
		
		input_patch = 		input_vol[ :,
						input_beginning[0]:input_end[0],
						input_beginning[1]:input_end[1],
						input_beginning[2]:input_end[2]]

		output_beginning = 	output_bounds[i][0]
		output_end = 		output_bounds[i][1]
		if verbose:
			print "Output Volume [{}] to [{}]".format(output_beginning, output_end)
			print ""

		# ACTUALLY RUNNING FORWARD PASS
		output_patch = net.forward( loa_as_continue(input_patch, dtype='float32') )
          
          
		output_vol[ :,
			output_beginning[0]:output_end[0],
			output_beginning[1]:output_end[1],
			output_beginning[2]:output_end[2]] = output_patch

	return output_vol

def save_output_volumes(output_volumes, prefix):
	'''
	Writes the resulting output volumes to disk according to the 
	output_prefix
	'''

	for i in range(len(output_volumes)):
		emio.imsave(output_volumes[i].astype('float32'), 
							"{}_{}.tif".format(prefix,i))

def test(input_patch, output_patch_shape, net):
	'''Silently generates an output patch for a single input patch'''

	return generate_output_volume(input_patch, output_patch_shape,
				net, verbose=False)

def config_forward_pass( config_filename, verbose=True ):
	'''
	Derives full forward pass for all samples specified within 
	a config file
	'''

	# parameters
	config, params = front_end.parser( config_filename )

	# read image stacks
	# Note: preprocessing included within CSamples
	# See CONSTANTS section above for optionname values
	sampler = front_end.CSamples( params[range_optionname], config, params ) 
	input_volumes = sampler.volume_dump()

	output_volumes = []

	output_patch_shape = params[outsz_optionname]

	# load network
	net = netio.load_network(params[net_optionname], 
					params[specfile_optionname], 
					params[outsz_optionname], 
					params[threads_optionname])

	# generating output volumes for each input
	for input_vol in input_volumes:

		output_volumes.append(
			generate_output_volume(input_vol, output_patch_shape, net,
				verbose=verbose)
			)

	return output_volumes

def main( config_filename ):
	'''
	Script functionality - runs config_forward_pass and saves the
	output volumes
	'''

	output_volumes = config_forward_pass( config_filename, verbose=True )

	print "Saving Output Volumes..."
	config, params = front_end.parser( config_filename )
	save_output_volumes( output_volumes, params[output_prefix_optionname] )

if __name__ == '__main__':

    from sys import argv
    if len(argv)>1:
        main( argv[1] )
    else:
        main('config.cfg')
