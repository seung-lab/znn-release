
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython
#@cython.boundscheck(False) # turn of bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef list twod_neighbors(int i, int j, int k, 
	int si, int sj, int sk):
	'''Returns a list of 4-connectivity neighbors within a 2d image'''

	# Init
	cdef list res = []
	cdef int[3] orig = (i, j, k)
	cdef int[3] shape = (si, sj, sk)

	#Index order: z,y,x
	cdef int dim
	cdef int[2] dims = (1,2)
	cdef int[3] candidate
	for dim in dims:

		candidate = (orig[0], orig[1], orig[2])
		candidate[dim] = candidate[dim] + 1

		if candidate[dim] < shape[dim]:
			res.append(candidate)

		candidate = (orig[0], orig[1], orig[2])
		candidate[dim] = candidate[dim] - 1

		if 0 <= candidate[dim]:
			res.append(candidate)

	#Using tuples for easier indexing
	return res

cdef bint contains(DTYPE_t[:] arr, int length, DTYPE_t value):
	'''Faster test for "{value} in {arr}" within cython'''

	cdef int i = 0
	while i < length:
		if arr[i] == value:
			return True
		i += 1
	return False

cpdef np.ndarray[DTYPE_t, ndim=3] relabel_volume(np.ndarray[DTYPE_t, ndim=3] vol):
	'''MAIN INTERFACE FUNCTION
	Relabels a 0/1 volume to have a separate
	border class'''

	s0 = vol.shape[0]
	s1 = vol.shape[1]
	s2 = vol.shape[2]

	#Init
	cdef np.ndarray[DTYPE_t, ndim=3] res = np.empty((s0, s1, s2), dtype=DTYPE)
	cdef int z,y,x, num_neighbors
	cdef list neighbor_indices
	cdef DTYPE_t[4] neighbor_values = (0,0,0,0)

	for z in xrange(s0):
		for y in xrange(s1):
			for x in xrange(s2):

				#Find indices of neighbors
				neighbor_indices = twod_neighbors(z,y,x, s0,s1,s2)
				#Find their values
				num_neighbors = len(neighbor_indices)

				for i in xrange(num_neighbors):
					neighbor_values[i] = vol[
						neighbor_indices[i][0],
						neighbor_indices[i][1],
						neighbor_indices[i][2]]

				#Assign new label according to collection of neighbor
				# values
				if contains(neighbor_values, num_neighbors, 1):
					if contains(neighbor_values, num_neighbors, 0) and vol[z,y,x] == 0:
						#border
						res[z,y,x] = 1
						continue
					else:
						#intra
						res[z,y,x] = 2
						continue
				else:
					#extra
					res[z,y,x] = 0
					continue

	return res
