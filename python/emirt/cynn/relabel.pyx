# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:40:02 2015

@author: jingpeng
"""
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)
#@cython.nonecheck(False)

DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

#%% depth first search
cdef dfs(np.ndarray[np.uint32_t, ndim=3] seg,\
        np.ndarray[np.uint32_t, ndim=3] seg2,\
        np.ndarray[np.uint8_t,  ndim=3, cast=True] mask, \
        np.uint32_t relid, \
        np.uint32_t label, \
        int z, int y, int x):
    cdef list seeds = []
    seeds.append((z,y,x))
    while seeds:
        z,y,x = seeds.pop()
        seg2[z,y,x] = relid
        mask[z,y,x] = True
        
        # 3D relabelling
        if z+1<seg.shape[0] and seg[z+1,y,x] == label and not mask[z+1,y,x] :
            seeds.append((z+1,y,x))
        if z-1>=0    and seg[z-1,y,x] == label and not mask[z-1,y,x] :
            seeds.append((z-1,y,x))
        if y+1<seg.shape[1] and seg[z,y+1,x] == label and not mask[z,y+1,x] :
            seeds.append((z,y+1,x))
        if y-1>=0    and seg[z,y-1,x] == label and not mask[z,y-1,x] :
            seeds.append((z,y-1,x))          
        if x+1<seg.shape[2] and seg[z,y,x+1] == label and not mask[z,y,x+1] :
            seeds.append((z,y,x+1))
        if x-1>=0    and seg[z,y,x-1] == label and not mask[z,y,x-1] :
            seeds.append((z,y,x-1))       
    return seg2, mask

#%% relabel by connectivity analysis
cpdef np.ndarray[DTYPE_t, ndim=3] relabel1N(np.ndarray[DTYPE_t, ndim=3] seg):
    print 'relabel by connectivity analysis ...'
    # masker for visiting
    cdef np.ndarray[np.uint8_t,    ndim=3, cast=True] mask 
    mask = (seg==0)
    
    sz = seg.shape[0]
    sy = seg.shape[1]
    sx = seg.shape[2]

    cdef np.ndarray[DTYPE_t, ndim=3] seg2 = np.zeros((sz, sy, sx), dtype=DTYPE)   # change to np.zeros ?
    # relabel ID
    cdef np.uint32_t relid = 0
    cdef np.uint32_t z,y,x
    for z in xrange(sz):
        for y in xrange(sy):
            for x in xrange(sx):
                if mask[z,y,x]:
                    continue
                relid += 1
                # flood fill
                seg2, mask = dfs(seg, seg2, mask, relid, seg[z,y,x], z,y,x)
    print "number of segments: {}-->{}".format( np.unique(seg).shape[0], np.unique(seg2).shape[0] )
    return seg2

def add_item(dict con, frozenset key, float value):
    if (not con.has_key(key)) or con[key]<value:
        con[key] = value
    return con
#%% estimate the connection probability, construct the dendrogram
def build_region_graph(np.ndarray[np.uint32_t, ndim=3] seg,\
                       np.ndarray[np.float32_t,  ndim=4] affin):
    cdef dict con = {}
    cdef int Nz = seg.shape[0]
    cdef int Ny = seg.shape[1]
    cdef int Nx = seg.shape[2]
    cdef int z,y,x
    for z in xrange(1, Nz):
        for y in xrange(1, Ny):
            for x in xrange(1, Nx):
                if seg[z,y,x]:
                    if seg[z-1,y,x] and seg[z,y,x]!=seg[z-1,y,x]:
                        con = add_item(con,frozenset([seg[z,y,x], seg[z-1,y,x]]), affin[2,z,y,x])
                    if seg[z,y-1,x] and seg[z,y,x]!=seg[z,y-1,x]:
                        con = add_item(con,frozenset([seg[z,y,x], seg[z,y-1,x]]), affin[1,z,y,x])
                    if seg[z,y,x-1] and seg[z,y,x]!=seg[z,y,x-1]:
                        con = add_item(con,frozenset([seg[z,y,x], seg[z,y,x-1]]), affin[0,z,y,x])
    return con