#!/usr/bin/env python
__doc__ = """

Nicholas Turner <nturner.stanford@gmail.com>
Jingpeng Wu <jingpeng.wu@gmail.com>, 2015

TODO- Better argument handling
- Fix single-volume output functionality

"""

from sys import argv

import numpy as np

from emirt import emio

#import pyznn
import front_end, netio
#import train_nt

def correct_output_patch_shape( output_patch_config_shape, net ):
	'''Returns a 4d version of the output shape array. Always replaces
	the 4th dimension with the num_output_vols specified

	Doubt this is necessary anymore'''

	num_output_vols = net.get_output_num()

	if output_patch_config_shape.size == 4:

		return np.hstack( (num_output_vols, output_patch_config_shape[1:]) )

	elif output_patch_config_shape.size == 3:

		return np.hstack( (num_output_vols, output_patch_config_shape) )

def input_patch_shape(output_patch_shape, fov):
	'''Determines the size of the input patch to feed into the network'''
	if output_patch_shape.size == 3:
		res = output_patch_shape + fov - 1
	else: #len == 4
		return np.hstack( (output_patch_shape[0],
							output_patch_shape[-3:] + fov - 1) )

def output_vol_shape(input_vol_shape, net):
	'''Derives the resulting shape of the full volume returned by the forward pass'''
	fov = np.asarray(net.get_fov())
	num_output_vols = net.get_output_num()

	return np.hstack( (num_output_vols, input_vol_shape[-3:] - fov + 1 ) )

def num_output_patches(output_vol_shape, output_patch_shape):
	'''One way to derive the number of output patches in the resulting volume.
	Mostly used for unit testing at this point

	Restricts calculation to 3d shape'''

	#3d Shape restriction
	output_vol_shape = output_vol_shape[-3:]
	output_patch_shape = output_patch_shape[-3:]

	# # per axis  = dim length / patch length (rounded up)
	num_per_axis = np.ceil(output_vol_shape / output_patch_shape.astype(np.float64))
	return int(np.prod(num_per_axis))

def patch_bounds(input_vol_width, output_patch_width, fov_width):
	'''Returns the bounds of one axis for a given input vol width and 
	output patch width'''

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

def input_patch_bounds(input_vol_shape, output_patch_shape, fov):
	'''Finds the bounds for each input patch given the input volume shape,
	the fov, and the output patch shape

	Restricts calculation to 3d shape'''

	#3d Shape restriction
	input_vol_shape = input_vol_shape[-3:]
	output_patch_shape = output_patch_shape[-3:]

	#Can be decomposed into a similar problem for each axis
	z_bounds = patch_bounds(input_vol_shape[0], output_patch_shape[0], fov[0])
	y_bounds = patch_bounds(input_vol_shape[1], output_patch_shape[1], fov[1])
	x_bounds = patch_bounds(input_vol_shape[2], output_patch_shape[2], fov[2])

	#And then recombined
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
	
def output_patch_bounds(output_vol_shape, output_patch_shape):
	'''Finds the bounds for each output patch given the output volume shape
	and the output patch shape'''
	return input_patch_bounds(output_vol_shape, 
				output_patch_shape, 
				# the 1 vector cancels out the fov contribution
				# within the patch_bounds function
				np.ones(output_patch_shape.shape).astype('uint32'))

def generate_output_volume(input_vol, output_patch_shape, net, verbose=True):
	'''Generates a full output volume for a given input volume - the main 
	functionality of the module

	shape of the output patch may be specified in 3d or 4d, but we'll
	overwrite any 4d specification with the output size of the net anyway'''

	#Making the input volume 4d
	if len(input_vol.shape) == 3:
		input_vol = input_vol.reshape( np.hstack((1,input_vol.shape)) )

	#Init
	output_vol = np.empty( 
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

	for i in xrange(len( input_bounds )):

		if verbose:
			print "Output patch #{}:".format(i+1) # i is just an index

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

		# ACTUALLY RUNNING FORWARD PASS
		#  Debug version to test indexing
		#  output_patch = np.zeros( output_patch_shape ) #Debug
		output_patch = net.forward( np.ascontiguousarray(input_patch) ).astype('float32')

		output_vol[ :,
			output_beginning[0]:output_end[0],
			output_beginning[1]:output_end[1],
			output_beginning[2]:output_end[2]] = output_patch

	return output_vol

def save_output_volumes(output_volumes, prefix):

	for i in range(len(output_volumes)):
		emio.imsave(output_volumes[i].astype('float32'), 
							"{}.{}.tif".format(prefix,i))

def test(input_patch, output_patch_shape, net):
	'''Silently generates an output patch for a single input patch'''

	return generate_output_volume(input_patch, output_patch_shape,
				net, verbose=False)

def main( config_filename ):
	'''Script functionality'''

	# parameters
	config, params = front_end.parser( config_filename )

	# read image stacks
	sampler = front_end.CSamples( params['forward_range'], config, params ) #preprocessing included here
	input_volumes = sampler.volume_dump()
	output_volumes = []

	output_patch_shape = params['forward_outsz']


	# load network
	# Debug - random network
	# net = train_nt.initialize_network( params )
	net = netio.load_network(params['forward_net'], 
					params['fnet_spec'], 
					params['forward_outsz'], 
					params['num_threads'])


	# generating output volumes for each input
	for input_vol in input_volumes:

		output_volumes.append(
			generate_output_volume(input_vol, output_patch_shape, net)
			)

	# saving
	print "Saving Output Volumes..."
	save_output_volumes(output_volumes, params['output_prefix'])
	print "Done"

if __name__ == '__main__':

	config_filename = argv[1]

	main( config_filename )
