#!/usr/bin/env cython

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
import scipy.sparse as sp
cimport cython
cimport numpy as np
#@cython.boundscheck(False) # turn of bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)

DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

cpdef overlap_matrix(
	np.ndarray[DTYPE_t, ndim=3] seg1, 
	np.ndarray[DTYPE_t, ndim=3] seg2):
	'''Calculates the overlap matrix between two segmentations of a volume'''

	cdef DTYPE_t seg1max = np.max(seg1)
	cdef DTYPE_t seg2max = np.max(seg2)

	cdef int num_segs1 = seg1max + 1 #+1 accounts for '0' segment
	cdef int num_segs2 = seg2max + 1

	#Representing the sparse overlap matrix as row/col/val arrays
	cdef np.ndarray[DTYPE_t] om_row, om_col, om_vals
	om_row = np.zeros(seg1.size, dtype=DTYPE)
	om_col = np.zeros(seg1.size, dtype=DTYPE)
	om_vals = np.ones(seg1.size, dtype=DTYPE) #value for now will always be one


	#Init
	sz = seg1.shape[0]
	sy = seg1.shape[1]
	sx = seg1.shape[2]

	cdef int z,y,x
	cdef int idx = 0
	cdef DTYPE_t seg1val, seg2val

	#loop over voxels
	for z in xrange(sz):
		for y in xrange(sy):
			for x in xrange(sx):
				om_row[idx] = seg1[z,y,x]
				om_col[idx] = seg2[z,y,x]

				idx += 1

	return sp.coo_matrix((om_vals, (om_row, om_col)), shape=(num_segs1, num_segs2)).tocsr()
