#!/usr/bin/env python
__doc__ = """
Relavitvely Quick Conversion of ZNN volume files to hdf5 file format

 This module transfers 3d channel data volumes, and 4d affinity graph
files (or any 4d output file), to hdf5 file format. The channel data 
is cropped to the 3d shape of one of the affinity volumes before 
conversion. Cropping takes evenly from both sides, in line with the 
loss of resolution common with convolutional nets.

Inputs:

	-Network Output Filename
	-Channel Data Filename
	-Output Filename 

Main Outputs:

	-Network Output HDF5 File ("{output_filename}")
	-Channel Data HDF5 File ("channel_{output_filename}")

Nicholas Turner, June 2015
"""

import h5py
import argparse
import numpy as np
from os import path
from vol_utils import crop, norm

import emio

def write_channel_file(data, filename, dtype='float32'):
	'''Placing the cropped channel data within an hdf5 file'''

	f = h5py.File(filename, 'w')

	dset = f.create_dataset('/main', tuple(data.shape), dtype=dtype)

	#Saving a NORMALIZED version of the data (0<=d<=1)
	dset[:,:,:] = norm(data.astype(dtype))

	f.close()

def write_affinity_file(data, filename, dtype='float32'):
	'''Placing the affinity graph within an hdf5 file dataset of 3d size
	specified by shape, and the number of volumes equal to the input data'''

	f = h5py.File(filename, 'w')

	dset = f.create_dataset('/main', tuple(data.shape), dtype=dtype) 
	
	#Saving data
	dset[:, :,:,:] = data.astype(dtype)

	f.close()
	

def main(net_output_filename, image_filename, output_filename):

	print "Importing data..."
	net_output = emio.znn_img_read(net_output_filename)
	image = emio.znn_img_read(image_filename)

	print "Cropping channel data..."
	#cropping the channel data to the 3d shape of the affinity graph
	cropped_image = crop(image, net_output.shape[-3:])

	image_outname = 'channel_{}'.format(path.basename(output_filename))

	print "Writing network output file..."
	write_affinity_file(net_output, output_filename)
	print "Writing image file..."
	write_channel_file(cropped_image, image_outname)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('net_output_filename',
		help='net affinity output')
	parser.add_argument('image_filename',
		help='image (channel) filename')
	parser.add_argument('output_filename')

	args = parser.parse_args()

	main(args.net_output_filename,
		 args.image_filename,
		 args.output_filename)
