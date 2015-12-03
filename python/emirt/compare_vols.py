#!/usr/bin/env python

__doc__ = '''
ZNN Volume Browser

 This module displays volumes as stacks of slices in the z-direction.

 If the volumes are of different sizes, then each volume is padded by 
 zeroes until the shapes match, with the exception of the z-direction.
 In this case, the volumes are truncated to match in number of slices.
 Regarding zero-padding in x and y, the padding adds evenly from both sides,
 in line with the loss of resolution common with convolutional nets.

 Once the volumes are loaded into the browser, they can be navigated by
 the following keyboard commands:

 up arrow - move up a slice
 down arrow - move down a slice
 [1-9] - select a plotted volume
 'c' - display a random coloring for the selected plot

Inputs:

	-Filenames of the volumes to display


Nicholas Turner, June 2015
'''
import h5py
import numpy as np
import emio, show
from sys import argv

def load_data(output_fname):

	if 'h5' not in output_fname:
		vol = emio.znn_img_read(output_fname)

		if len(vol.shape) > 3:
			if vol.shape[0] > 2: #multiclass output
				vol = vol[0,:,:,:]
				# vol = np.argmax(vol, axis=0)
			else: #binary output
				vol = vol[0,:,:,:]
	else:
		f = h5py.File(output_fname)
		vol = f['/main'] 

	return vol

def main(fname_list):
	'''Loads data, starts comparison'''

	vols = [load_data(fname) for fname in fname_list]

	com = show.CompareVol(vols)
	com.vol_compare_slice()

if __name__ == '__main__':

	main(argv[1:])
