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

def lbl_RGB2uint32( lbl ):
    # read the VAST output RGB images
    assert( lbl.dtype=='uint8' and lbl.shape[3]==3 )
    lbl = lbl.astype('uint32')
    lbl = lbl[:,:,:,0]*256*256 + lbl[:,:,:,1]*256 + lbl[:,:,:,2]
    return lbl

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

def mark_bd(seg):
    unique, indices, counts = np.unique(seg, return_index=True, return_counts=True)
    # binary affinity graphs
    inds = indices[counts==1]
    seg2 = seg.flatten()
    seg2[inds] = 0
    seg = seg2.reshape( seg.shape )
    return seg

def bdm2aff( bdm, Dim = 2 ):
    """
    transform boundary map to affinity map
    currently only do 2D, Z affinity will always be 0!

    Parameters
    ----------
    bdm: 2D/3D numpy array, float, boundary map

    Returns
    -------
    affs: affinity map, zyx direction
    """
    # only support 2D now, could be extended
    assert( Dim==2 )
    # always make bdm 3D
    if bdm.ndim == 2:
        bdm = bdm.reshape( (1,)+ bdm.shape )
    assert bdm.ndim == 3

    # initialization
    affs_shape = (3,) + bdm.shape
    affs = np.zeros( affs_shape, bdm.dtype)

    # get y affinity
    for z in xrange( bdm.shape[0] ):
        for y in xrange( 1, bdm.shape[1] ):
            for x in xrange( bdm.shape[2] ):
                affs[1,z,y,x] = min( bdm[z,y,x], bdm[z, y-1, x] )

    # get x affinity
    for z in xrange( bdm.shape[0] ):
        for y in xrange( bdm.shape[1] ):
            for x in xrange( 1, bdm.shape[2] ):
                affs[0,z,y,x] = min( bdm[z,y,x], bdm[z, y,   x-1] )

    return affs


def aff2seg( affs, threshold=0.5 ):
    """
    get segmentation from affinity graph using union-find algorithm.
    tsz weighted quick union with path compression:
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
    # should be 4 dimension
    assert affs.ndim==4

    # get affinity graphs, copy the array to avoid changing of raw affinity graph
    xaff = np.copy( affs[0,:,:,:] )
    yaff = np.copy( affs[1,:,:,:] )
    zaff = np.copy( affs[2,:,:,:] )

    # initialize segmentation with individual label of each voxel
    vids = np.arange(xaff.size).reshape( xaff.shape )
    # use disjoint sets
    import domains
    djset = domains.CDisjointSets( xaff.size )

    # z affnity
    for z in xrange( 1, zaff.shape[0] ):
        for y in xrange( zaff.shape[1] ):
            for x in xrange( zaff.shape[2] ):
                if zaff[z,y,x]>threshold:
                    vid1 = vids[z,   y, x]
                    vid2 = vids[z-1, y, x]
                    rid1 = djset.find_root( vid1 )
                    rid2 = djset.find_root( vid2 )
                    djset.join( rid1, rid2 )

    # y affinity
    for z in xrange( yaff.shape[0] ):
        for y in xrange( 1, yaff.shape[1] ):
            for x in xrange( yaff.shape[2] ):
                if yaff[z,y,x]>threshold:
                    vid1 = vids[z, y,   x]
                    vid2 = vids[z, y-1, x]
                    rid1 = djset.find_root( vid1 )
                    rid2 = djset.find_root( vid2 )
                    djset.join( rid1, rid2 )
    # x affinity
    for z in xrange( xaff.shape[0] ):
        for y in xrange( xaff.shape[1] ):
            for x in xrange( 1, xaff.shape[2] ):
                if xaff[z,y,x]>threshold:
                    vid1 = vids[z, y, x  ]
                    vid2 = vids[z, y, x-1]
                    rid1 = djset.find_root( vid1 )
                    rid2 = djset.find_root( vid2 )
                    djset.join( rid1, rid2 )

    # get current segmentation
    # note that the boundary voxels have separet segment id
    seg = djset.get_seg().reshape( xaff.shape )

    # remove the boundary segments
    seg = mark_bd(seg)

    return seg

def seg2aff( lbl, affs_dtype='float32' ):
    """
    transform labels to true affinity.

    Parameters
    ----------
    lbl : 3D uint32 array, manual label volume.

    Returns
    -------
    aff : 4D float array, affinity graph.
    """
    if lbl.ndim ==2 :
        lbl = lbl.reshape( (1,)+lbl.shape )
    elif lbl.ndim ==4:
        assert lbl.shape[0]==1
        lbl = lbl.reshape( lbl.shape[1:] )
    # the 3D volume number should be one
    assert lbl.ndim==3

    affs_shape = (3,) + lbl.shape

    affs = np.zeros( affs_shape , dtype= affs_dtype  )

    # z affinity
    for z in xrange( 1, affs.shape[1] ):
        for y in xrange( affs.shape[2] ):
            for x in xrange( affs.shape[3] ):
                if (lbl[z,y,x]==lbl[z-1,y,x]) and lbl[z,y,x]>0 :
                    affs[2,z,y,x] = 1.0

    # y affinity
    for z in xrange( affs.shape[1] ):
        for y in xrange( 1, affs.shape[2] ):
            for x in xrange( affs.shape[3] ):
                if (lbl[z,y,x]==lbl[z,y-1,x]) and lbl[z,y,x]>0 :
                    affs[1,z,y,x] = 1.0

    # x affinity
    for z in xrange( affs.shape[1] ):
        for y in xrange( affs.shape[2] ):
            for x in xrange( 1, affs.shape[3] ):
                if (lbl[z,y,x]==lbl[z,y,x-1]) and lbl[z,y,x]>0 :
                    affs[0,z,y,x] = 1.0

    return affs

def bdm2seg_2D( bdm, threshold=0.5, is_relabel=True ):
    """
    transform 2D boundary map to segmentation using connectivity analysis.

    Parameters
    ----------
    bdm: 2D float array with value [0,1], boundary map with black boundary
    threshold: the binarize threshold
    is_relabel: whether relabel the segment id to 1-N

    Return
    ------
    seg: 2D uint32 array, segmentation
    """
    if bdm.ndim==3 and bdm.shape[0]==1:
        bdm = bdm.reshape((bdm.shape[1], bdm.shape[2]))
    # make sure that this is a 2D array
    assert( bdm.ndim==2 )
    # binarize the volume using threshold
    bmap = ( bdm>threshold )

    # segmentation initialized with 1-N,
    # the 0 is left for boundaries
    seg = np.arange( 1, bmap.size+1 )
    # segment ids
    ids = np.copy(seg).reshape( bmap.shape )
    # tree size of union-find
    tsz = np.ones( bdm.size )

    # traverse each connectivity along x axis
    for y in xrange(bdm.shape[0]):
        for x1 in xrange( bdm.shape[1]-1 ):
            x2 = x1+1
            if bmap[y,x1] and bmap[y,x2]:
                # the id of pixel
                id1 = ids[y,x1]
                id2 = ids[y,x2]
                # find the root
                r1, seg = find_root(id1, seg)
                r2, seg = find_root(id2, seg)
                # union the two tree
                seg, tsz = union_tree(r1, r2, seg, tsz)

    # traverse each connectivity along y axis
    for x in xrange(bdm.shape[1]):
        for y1 in xrange( bdm.shape[0]-1 ):
            y2 = y1+1
            if bmap[y1,x] and bmap[y2,x]:
                # the id of pixel
                id1 = ids[y1,x]
                id2 = ids[y2,x]
                # find the root
                r1, seg = find_root(id1, seg)
                r2, seg = find_root(id2, seg)
                # union the two tree
                seg, tsz = union_tree(r1, r2, seg, tsz)

    # relabel all the trees to root id
    for k in xrange(seg.size):
        root_ind, seg = find_root(seg[k], seg)
        seg[k] = root_ind
    # reshape to original shape
    seg = seg.reshape(bdm.shape)
    # remove the boundary segments
    seg = mark_bd(seg)

    # relabel the segment id to 1-N
    if is_relabel:
        seg = relabel_1N(seg)
    return seg

def relabel_1N(seg):
    """
    relabel the segment id to 1-N
    """
    # find the id mapping
    ids1 = np.unique(seg)
    mp = dict()
    for i in xrange( len(ids1) ):
        mp[ ids1[i] ] = i+1

    # replace the segment ids
    for k in xrange(seg.size):
        seg.flat[k] = mp[ seg.flat[k] ]
    return seg

def bdm2seg(bdm, threshold=0.5, is_label=True):
    """
    transform 3D boundary map to segmentation
    """
    if bdm.ndim==4 and bdm.shape[0]==1:
        bdm = bdm.reshape((bdm.shape[1], bdm.shape[2], bdm.shape[3]))
    # make sure that this is a 3D array
    assert(bdm.ndim==3)
    # initialize the segmentation
    seg = np.empty(bdm.shape, dtype='uint32')
    # the maximum id of previous section
    maxid = 0
    for z in xrange(bdm.shape[0]):
        seg2d = bdm2seg_2D(bdm[z,:,:], threshold, is_relabel=True)
        seg[z,:,:] = maxid + seg2d
        # update the maximum segment id
        maxid = np.max(seg[z,:,:])
    return seg
