#!/usr/bin/env python
__doc__ = """

Nicholas Turner <nturner.stanford@gmail.com>
Jingpeng Wu <jingpeng.wu@gmail.com>, 2015

TODO- Better argument handling

"""

from sys import argv

import numpy as np

from emirt import emio

import pyznn
import front_end, front_end_io

def load_output_patch_shape( output_patch_config_shape ):
	'''Returns a 4d version of the output shape array. Prints a warning
	message if output shape is 3d, and assumes affinity output'''

	if output_patch_config_shape.size == 4:

		return output_shape_array

	elif output_patch_config_shape.size == 3:

		print "WARNING: only 3 output patch dimensions specified"
		print "Assuming 3-volume output"

		return np.hstack( (3, output_patch_config_shape) )

def input_patch_shape(output_patch_shape, fov):
	'''Determines the size of the input patch to feed into the network'''
	if output_patch_shape.size == 3:
		res = output_patch_shape + fov - 1
	else: #len == 4
		return np.hstack( (output_patch_shape[0],
							output_patch_shape[-3:] + fov - 1) )

def output_vol_shape(fov, input_vol_shape, num_vols):
	'''Derives the resulting shape of the full volume returned by the forward pass'''
	return np.hstack( (num_vols, input_vol_shape[-3:] - fov + 1 ) )

def num_output_patches(output_vol_shape, output_patch_shape):
	'''One way to derive the number of output patches in the resulting volume.
	Mostly used for unit testing at this point'''
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
	the fov, and the output patch shape'''

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

def generate_output_volume(input_vol, output_patch_shape_4d, fov, net, verbose=True):
	'''Generates a full output volume for a given input volume - the main 
	functionality of the module'''

	input_patch_shape_3d = input_vol.shape[-3:]
	output_patch_shape_3d = output_patch_shape_4d[-3:]

	#Init
	output_vol = np.empty( 
			output_vol_shape(fov, input_vol.shape, output_patch_shape_4d[0])
			# output_patch_shape_4d[0] = # output volumes for the net
			).astype('float32')

	#Derive bounds of input and output patches
	input_bounds = input_patch_bounds( input_patch_shape_3d, 
						output_patch_shape_3d,
						fov )
	output_bounds = output_patch_bounds( output_vol.shape[-3:],
						output_patch_shape_3d )

	#Stupidity checks (so far so good!)
	assert num_output_patches( output_vol.shape[-3:], output_patch_shape_3d ) == len(output_bounds)
	assert len( input_bounds ) == len( output_bounds )

	for i in xrange(len( input_bounds )):

		if verbose:
			print "Output patch #{}:".format(i+1) # i is just an index

		input_beginning = 	input_bounds[i][0]
		input_end = 		input_bounds[i][1]
		if verbose:
			print "Input Volume [{}] to [{}]".format(input_beginning, input_end)
		
		input_patch = 		input_vol[
						input_beginning[0]:input_end[0],
						input_beginning[1]:input_end[1],
						input_beginning[2]:input_end[2]]
		#4dimensionalizing
		input_patch = input_patch.reshape( np.hstack((1,input_patch.shape)) )

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

		emio.znn_img_save(output_volumes[i].astype('double'), 
							"{}.{}".format(prefix,i))

def test(input_patch, output_patch_shape, net, fov=None):
	'''Generates an output patch for a single input patch'''

	if fov is None:
		# fov = (input_patch.shape - output_patch_shape + 1)[-3:]
		fov = np.asarray(net.get_fov())

	return generate_output_volume(input_patch, output_patch_shape,
			 fov, net, verbose=False)

def main( config_filename ):
	'''Script functionality'''

	# parameters
	(global_params, 
     train_params, 
     forward_params) = front_end.parser( config_filename )

	# read image stacks
	input_volumes = front_end.read_tifs(forward_params['ffwds'])
	output_volumes = []

	output_patch_shape_4d = load_output_patch_shape(forward_params['outsz'])


	# load network
	# Debug - random network
	# net = train_nt.initialize_network(train_params, global_params)
	net = front_end_io.load_network(global_params['fnet'], 
					global_params['fnet_spec'], 
					forward_params['outsz'], 
					global_params['num_threads'])

	fov = np.asarray(net.get_fov())
	print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])

	# generating output volumes for each input
	for input_vol in input_volumes:

		output_volumes.append(
			generate_output_volume(input_vol, output_patch_shape_4d, fov, net)
			)

	# saving
	save_output_volumes(output_volumes, forward_params['out_prefix'])

if __name__ == '__main__':

	config_filename = argv[1]

	main( config_filename )
