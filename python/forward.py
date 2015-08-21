#!/usr/bin/env python
__doc__ = """

Nicholas Turner <nturner.stanford@gmail.com>
Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

from emirt import emio
import numpy as np
import pyznn
import front_end
import front_end_io

# get input patch size

def output_vol_shape(fov, input_vol_shape):
	return input_vol_shape - fov + 1

def num_output_patches(output_vol_shape, output_patch_shape):
	# # per axis  = dim length / patch length (rounded up)
	num_per_axis = np.ceil(output_vol_shape / output_patch_shape.astype(np.float64))
	return int(np.prod(num_per_axis))

def input_patch_shape(output_patch_shape, fov):
	return output_patch_size + fov - 1

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
	return input_patch_bounds(output_vol_shape, output_patch_shape, output_patch_shape)

def generate_output_volume(input_vol, output_patch_shape, fov):
	'''Derives a full output volume for a given input volume'''

	#Init
	output_vol = np.empty( output_vol_shape(fov, input_vol.shape) ).astype('float32')

	#Derive bounds of input and output patches
	input_bounds = input_patch_bounds( input_vol.shape, 
									output_patch_shape,
									fov )
	output_bounds = output_patch_bounds( output_vol.shape,
									output_patch_shape )

	#Stupidity check and debug output
	# print "Input Bounds"
	# print input_bounds
	# print "Output Bounds"
	# print output_bounds
	# print "# Output Patches"
	# print num_output_patches (output_vol.shape, fov)
	assert num_output_patches( output_vol.shape, fov ) == len(output_bounds)
	assert len( input_bounds ) == len( output_bounds )

	for i in xrange(len( input_bounds )):
		print "Output patch: {}".format(i)

		input_beginning = 	input_bounds[i][0]
		input_end = 		input_bounds[i][1]
		print "Input Volume [{}] to [{}]".format(input_beginning, input_end)
		input_patch = 		input_vol[
								input_beginning[0]:input_end[0],
								input_beginning[1]:input_end[1],
								input_beginning[2]:input_end[2]]

		output_beginning = 	output_bounds[i][0]
		output_end = 		output_bounds[i][1]
		print "Output Volume [{}] tp [{}]".format(output_beginning, output_end)

		# CURRENTLY NOT FUNCTIONAL (waiting for znn back end changes)
		output_patch = np.zeros( output_patch_shape) #Debug
		# output_patch = net.forward( np.ascontiguousarray(input_patch) ).astype('float32')

		output_vol[
			output_beginning[0]:output_end[0],
			output_beginning[1]:output_end[1],
			output_beginning[2]:output_end[2]] = output_patch

	return output_vol

def save_output_volumes(output_volumes, prefixes):

	for i in range(len(output_volumes)):

		emio.znn_img_save(output_volumes[i], prefixes[i])

def test(input_patch, fov):

	output_patch_shape = input_patch.shape - fov + 1

	return generate_output_volume(input_patch, output_patch_shape, fov)

def main():

	# parameters
	gpars, tpars, fpars = front_end.parser( 'config.cfg' )

	# read image stacks
	input_volumes = front_end.read_tifs(fpars['ffwds'])
	output_volumes = []

	output_patch_size = fpars["outsz"]
	fov = np.asarray(net.get_fov())
	print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])

	# load network
	net = front_end_io.load_network(gpars['fnet'], 
									gpars['fnet_spec'], 
									fpars['outsz'], 
									gpars['num_threads'])


	# generating output volumes for each input
	for input_vol in input_volumes:

		output_volumes.append(
			generate_output_volume(input_vol, output_patch_shape, fov)
			)

	# saving
	save_output_volumes(output_volumes, fpars['out_prefix'])
