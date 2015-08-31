# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:40:02 2015

@author: nicholasturner1
"""
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython
#@cython.boundscheck(False) # turn of bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.uint32_t DTYPE2_t

cpdef np.ndarray[DTYPE_t, ndim=3] x_affinity(np.ndarray[np.uint32_t, ndim=3] seg,
	DTYPE_t pos_edge_val,
	DTYPE_t neg_edge_val):

    cdef int sz, sy, sx
    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    sz -= 1
    sy -= 1
    sx -= 1

    cdef np.ndarray[DTYPE_t, ndim=3] affinity = np.zeros((sz, sy, sx), dtype=DTYPE)

    cdef int z, y, x
    cdef DTYPE2_t v1, v2
    for z in xrange(sz):
        for y in xrange(sy):
            for x in xrange(sx):
                v1 = seg[z+1,y+1,x]
                v2 = seg[z+1,y+1,x+1]

                if v1 == 0 and v2 == 0:
                    affinity[z,y,x] = neg_edge_val
                else:
                    if v1 - v2 == 0:
                        affinity[z,y,x] = pos_edge_val
                    else:
                        affinity[z,y,x] = neg_edge_val

    return affinity

cpdef np.ndarray[DTYPE_t, ndim=3] y_affinity(np.ndarray[np.uint32_t, ndim=3] seg,
	DTYPE_t pos_edge_val,
	DTYPE_t neg_edge_val):

    cdef int sz, sy, sx
    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    sz -= 1
    sy -= 1
    sx -= 1

    cdef np.ndarray[DTYPE_t, ndim=3] affinity = np.zeros((sz, sy, sx), dtype=DTYPE)

    cdef int z, y, x
    cdef DTYPE2_t v1, v2
    for z in xrange(sz):
        for y in xrange(sy):
            for x in xrange(sx):
                v1 = seg[z+1,y,x+1]
                v2 = seg[z+1,y+1,x+1]

                if v1 == 0 and v2 == 0:
                    affinity[z,y,x] = neg_edge_val
                else:
                    if v1 - v2 == 0:
                        affinity[z,y,x] = pos_edge_val
                    else:
                        affinity[z,y,x] = neg_edge_val

    return affinity

cpdef np.ndarray[DTYPE_t, ndim=3] z_affinity(np.ndarray[np.uint32_t, ndim=3] seg,
	DTYPE_t pos_edge_val,
	DTYPE_t neg_edge_val):

    cdef int sz, sy, sx
    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    sz -= 1
    sy -= 1
    sx -= 1

    cdef np.ndarray[DTYPE_t, ndim=3] affinity = np.zeros((sz, sy, sx), dtype=DTYPE)

    cdef int z, y, x
    cdef DTYPE2_t v1, v2
    for z in xrange(sz):
        for y in xrange(sy):
            for x in xrange(sx):
                v1 = seg[z,y+1,x+1]
                v2 = seg[z+1,y+1,x+1]

                if v1 == 0 and v2 == 0:
                    affinity[z,y,x] = neg_edge_val
                else:
                    if v1 - v2 == 0:
                        affinity[z,y,x] = pos_edge_val
                    else:
                        affinity[z,y,x] = neg_edge_val

    return affinity
