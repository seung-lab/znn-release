# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:40:08 2015

compile to speedup:
    cython -a volume_util.py
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o volume_util.so volume_util.c

@author: jingpeng
"""
import numpy as np
#from numba import autojit

#%% add boundary between connected regions
def add_boundary_im(im):
    Ni, Nj = im.shape
    im2 = np.copy(im)
    for i in range(1,Ni-1):
        for j in range(1,Nj-1):
            mat = im[i-1:i+2, j-1:j+2]
            nzi,nzj = mat.nonzero()
            if len(np.unique( mat[nzi,nzj] ))>1 :
                im2[i,j]=0
    return im2

def add_boundary_2D(vol):
    Nz,Ny,Nx = vol.shape 
    for z in range(Nz): 
        vol[z,:,:] = add_boundary_im(vol[z,:,:])
    return vol
    
def add_boundary_3D(vol, neighbor = 6):
    Nz,Ny,Nx = vol.shape 
    vol2 = np.copy(vol)
    for z in range(1,Nz-1):
        for y in range(1,Ny-1):
            for x in range(1,Nx-1):
                mat = vol[z-1:z+2,y-1:y+2,x-1:x+2]
                if neighbor == 6:
                     neighbor6 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,1,1,1,0, 1, 0, 0, 0,0, 0, 1, 0, 0, 0, 0]
                     neighbor6 = np.asarray(neighbor6).reshape(3,3,3)
                     mat = mat * neighbor6
                nzz,nzy,nzx = mat.nonzero()
                if mat[1,1,1]!=0 and len(np.unique( mat[nzz,nzy,nzx] )) > 1:
                    vol2[z,y,x] = 0
    # the first and last image
    vol2[0,:,:] = add_boundary_im(vol[0,:,:])
    vol2[Nz-1,:,:] = add_boundary_im(vol[Nz-1,:,:])
    return vol2


def crop(vol, target_shape):
    '''Currently only returns value of crop3d'''
    return crop3d(vol, target_shape)

def crop3d(vol, target_shape, round_up=None, pick_right=None):
    '''
    Crops the input 3d volume to fit to the target 3d shape

    round_up: Whether to crop an extra voxel in the case of an odd dimension
    difference
    pick_right: Whether to prefer keeping the earlier index voxel in the case of
    an odd dimension difference
    '''
    dim_diffs = np.array(vol.shape) - np.array(target_shape)

    #Error checking
    odd_dim_diff_exists = any([dim_diffs[i] % 2 == 1 for i in range(len(dim_diffs))])
    if odd_dim_diff_exists and round_up == None and pick_right == None:
        raise ValueError('Odd dimension difference between volume shape and target' + 
                         ' with no handling specified')

    if any([vol.shape[i] < target_shape[i] for i in range(len(target_shape))]	):
        raise ValueError('volume already smaller that target volume!')

    #Init
    margin = np.zeros(dim_diffs.shape)
    if round_up:
        margin = np.ceil(dim_diffs / 2.0).astype(np.int)

    #round_up == False || round_up == None
    elif pick_right != None: 
        #voxel selection option will handle the extra
        margin = np.ceil(dim_diffs / 2.0).astype(np.int)

    else: #round_up == None and pick_right == None => even dim diff
        margin = dim_diffs / 2

    zmin = margin[0]; zmax = vol.shape[0] - margin[0]
    ymin = margin[1]; ymax = vol.shape[1] - margin[1]
    xmin = margin[2]; xmax = vol.shape[2] - margin[2]

    #THIS SECTION NOT ENTITRELY CORRECT YET
    # DOESN'T TAILOR 'SELECTION' TO AXES WITH THE ODD DIM DIFFS
    if odd_dim_diff_exists and pick_right:

        zmax += 1; ymax += 1; xmax += 1

    elif odd_dim_diff_exists and pick_right != None: 
        #pick_right == False => pick_left
        
        zmin -= 1; ymin -= 1; xmin -= 1

    return vol[zmin:zmax, ymin:ymax, xmin:xmax]

def norm(vol):
    '''Normalizes the input volume to have values between 0 and 1
    (achieved by factor normalization to the max)'''
    vol = vol - np.min(vol.astype('float32'))
    vol = vol / np.max(vol)
    return vol

#@autojit(nopython=True)
def find_root(ind, seg):
    """
    quick find with path compression

    Parameters
    ----------
    ind:   index of node. start from 1
    seg:   segmenation ID, should be flat

    Return
    ------
    ind: root index of input node
    seg:  updated segmentation
    """
    path = list()
    while seg[ind-1]!=ind:
        path.append( ind )
        # get the parent index
        ind = seg[ind-1]
    # path compression
    for node in path:
        seg[node-1] = ind
    return (ind, seg)

#@autojit(nopython=True)
def union_tree(r1, r2, seg, tree_size):
    """
    union-find algorithm: tree_sizeed quick union with path compression

    Parameters
    ----------
    r1,r2:  index of two root nodes.
    seg:   the segmenation volume with segment id. this array should be flatterned.
    tree_size: the size of tree.

    Return
    ------
    seg:       updated segmentation
    tree_size:    updated tree_size
    """
    # merge small tree to big tree according to size
    if tree_size[r1-1] < tree_size[r2-1]:
        r1, r2 = r2, r1
    seg[r2-1] = r1
    tree_size[r1-1] = tree_size[r1-1] + tree_size[r2-1]
    return (seg, tree_size)

def mark_bd(seg):
    unique, indices, counts = np.unique(seg, return_index=True, return_counts=True)
    # binary affinity graphs    
    inds = indices[counts==1]
    seg2 = seg.flatten()
    seg2[inds] = 0
    seg = seg2.reshape( seg.shape )
    return seg   
    
def seg_affs( affs, threshold=0.5 ):
    """
    get segmentation from affinity graph using union-find algorithm.
    tree_size weighted quick union with path compression: 
    https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf

    Parameters:
    -----------
    affs:  4D array of affinity graph

    Returns:
    --------
    seg:   3D array, segmentation of affinity graph
    """
    if isinstance(affs, dict):
        assert(len(affs.keys())==1)
        affs = affs.values()[0]
        
    # get affinity graphs, copy the array to avoid changing of raw affinity graph
    xaff = np.copy( affs[2,:,:,:] )
    yaff = np.copy( affs[1,:,:,:] )
    zaff = np.copy( affs[0,:,:,:] )

    # get edges
    xedges = np.argwhere( xaff>threshold )
    yedges = np.argwhere( yaff>threshold )
    zedges = np.argwhere( zaff>threshold )

    # initialize segmentation with individual label of each voxel
    seg_shp = np.asarray( xaff.shape ) + 1
    N = np.prod( seg_shp ) 
    ids = np.arange(1, N+1).reshape( seg_shp )
    seg = np.copy( ids ).flatten()
    tree_size = np.ones( seg.shape ).flatten()
    # create edge pair
    for e in xedges:
        # get the index of connected nodes
        id1 = ids[e[0], e[1], e[2]]
        id2 = ids[e[0], e[1], e[2]+1]
        # union-find algorithm
        r1, seg = find_root(id1, seg)
        r2, seg = find_root(id2, seg)
        seg, tree_size = union_tree(r1, r2, seg, tree_size)
    for e in yedges:
        # get the index of connected nodes
        id1 = ids[e[0], e[1],   e[2]]
        id2 = ids[e[0], e[1]+1, e[2]]
        # union-find algorithm
        r1, seg = find_root(id1, seg)
        r2, seg = find_root(id2, seg)
        seg, tree_size = union_tree(r1, r2, seg, tree_size)
    for e in zedges:
        # get the index of connected nodes
        id1 = ids[e[0]  , e[1], e[2]]
        id2 = ids[e[0]+1, e[1], e[2]]
        # union-find algorithm
        r1, seg = find_root(id1, seg)
        r2, seg = find_root(id2, seg)
        seg, tree_size = union_tree(r1, r2, seg, tree_size)
    
    # relabel all the trees to root id
    for k in xrange(seg.size):
        root_ind, seg = find_root(seg[k], seg)
        seg[k] = root_ind
    seg = np.reshape(seg, seg_shp)
    
    # remove the boundary segments
    seg = mark_bd(seg)
    
    return seg