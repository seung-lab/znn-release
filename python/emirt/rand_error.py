#!/usr/bin/env python
__doc__ = '''
3D Rand Error Between ZNN Volumes

 This module computes the 3D Rand Error between
two ZNN volumes. It also saves a volume for each of the inputs,
which contains the connected components found by 4-connectivity analysis.

Inputs:

	-Network Output Filename
	-Label Filename
	-Threshold for the network output
	-Whether to save connected component volumes (opt) (flag)

Main Outputs:

	-Reports rand error via a print
	-Saves connected component volumes to disk

Nicholas Turner, Jingpeng Wu June 2015
'''

import timeit
import argparse
import numpy as np

import emio
from cynn import relabel, overlap_matrix
from os import path

def threshold_volume(vol, threshold=0.5):
	return (vol > threshold).astype('uint32')

def choose_two(n):
	# = (n * (n-1)) / 2.0, with fewer overflows
	return (n / 2.0) * (n-1)

vchoose_two = np.vectorize(choose_two)

def om_rand_error(om, merge_err=False, split_err=False):
	'''Calculates the rand error of an unnormalized (raw counts) overlap matrix

	Can also return split and/or merge error separately ASSUMING that the "ground-truth"
	segmentation is represented in the 2nd axis (columns)'''

	counts1 = om.sum(1)
	counts2 = om.sum(0)

	#float allows division
	N = float(counts1.sum())

	# True
	a_term = np.sum(vchoose_two(counts1))
	b_term = np.sum(vchoose_two(counts2))

	# Estimate
	# a_term = np.sum(np.square(counts1 / N))
	# b_term = np.sum(np.square(counts2 / N))

	# Yields overflow errors
	# a_term = np.sum(np.square(counts1)) / (N ** 2)
	# b_term = np.sum(np.square(counts2)) / (N ** 2)

	# True
	p_ij_vals = vchoose_two(np.copy(om.data))
	p_term = np.sum(p_ij_vals)

	# Estimate
	#p term requires a bit more work with sparse matrix
	# sq_vals = np.square(np.copy(om.data))
	# p_term = np.sum(sq_vals) / (N ** 2)

	total_pairs = choose_two(N)

	merge_error = (a_term - p_term) / total_pairs
	split_error = (b_term - p_term) / total_pairs

	full_error = (a_term + b_term - 2*p_term) / total_pairs

	# print "Merge Error: %g" % merge_error
	# print "Split Error: %g" % split_error
	# print "Full Error: %g" % full_error

	if split_err and merge_err:
		return full_error, merge_error, split_error
	elif split_err:
		return full_error, split_error
	elif merge_err:
		return full_error, merge_error
	else:
		return full_error

def seg_rand_error(seg1, seg2, merge_err=False, split_err=False):
	'''Higher-level function which handles computing the overlap matrix'''

	om = overlap_matrix.overlap_matrix(seg1, seg2)

	return om_rand_error(om, merge_err, split_err)

def seg_fr_rand_error(seg1, seg2, merge_err=False, split_err=False):
	'''Similar high-level function restricting calculation to seg2's foreground'''

	seg1_fr = seg1[seg2 != 0]
	seg2_fr = seg2[seg2 != 0]

	om = overlap_matrix.overlap_matrix1d(seg1_fr, seg2_fr)

	return om_rand_error(om, merge_err, split_err)

def get_re(vol, label, threshold=0.5, save=False):

	if len(vol.shape) > 3:
		if vol.shape[0] > 2:
			vol = np.argmax(vol, axis=0).astype('uint32')
		else:
			vol = vol[1,:,:,:]

			print "Thresholding output volume..."
			vol = threshold_volume(vol, threshold)

	print "Labelling connected components in volume..."
	start = timeit.default_timer()
	vol_cc = relabel.relabel1N(vol)
	end = timeit.default_timer()
	print "Labelling completed in %f seconds" % (end-start)
	print
	print "Labelling connected components in labels..."
	start = timeit.default_timer()
	label_cc = relabel.relabel1N(label.astype('uint32'))
	end = timeit.default_timer()
	print "Labelling completed in %f seconds" % (end-start)

	if save:
		print "Saving labelled connected components..."
		emio.znn_img_save(vol_cc.astype(float), 'cc_{}'.format(path.basename(vol_fname)))
		emio.znn_img_save(label_cc.astype(float), 'cc_{}'.format(path.basename(label_fname)))

	print
	print "Finding overlap matrix..."
	start = timeit.default_timer()
	om = overlap_matrix.overlap_matrix(vol_cc, label_cc)
	end = timeit.default_timer()
	print "Matrix Calculated in %f seconds" % (end-start)

	print "Calculating Rand Error..."
	RE = om_rand_error(om)

	print "Rand Error: "
	print RE

	return vol_cc

def main(vol_fname, label_fname, threshold=0.5, save=False):
    print "Loading Data..."
	vol = emio.imread(vol_fname)
	label = emio.imread(label_fname)

    get_re(vol, label, threshold, save)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('output_filename',
		help="Filename of the output image")
	parser.add_argument('label_filename',
		help="Filename of the labels for comparison")
	parser.add_argument('threshold',
		nargs='?',default=0.5, type=float,
		help="Threshold for generating binary image")
	parser.add_argument('-no_save',
		default=True, action='store_false')

	args = parser.parse_args()

	main(args.output_filename,
		 args.label_filename,
		 args.threshold,
		 args.no_save)
