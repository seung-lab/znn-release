#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np

def square_loss(props, lbls):
    """
    compute square loss

    Parameters:
    props:   list of forward pass output
    lbls:    list of ground truth labeling
    Return:
    err:    cost energy
    cls:    classification error
    grdts:  list of gradient volumes
    """
    grdts = list()
    cls = 0
    err = 0
    for prop, lbl in zip( props, lbls ):
        cls = cls + float(np.count_nonzero( (prop>0.5)!= lbl ))

        grdt = prop.copy()
        grdt = grdt.astype('float32') - lbl.astype('float32')
        err = err + np.sum( grdt * grdt ).astype('float32')
        grdt = grdt * 2
        grdt = np.ascontiguousarray(grdt, dtype='float32')
        grdts.append( grdt )
    cls = cls / float( len(props) )
    err = err / float( len(props) )
    return (err, cls, grdts)

def binomial_cross_entropy(props, lbls):
    """
    compute binomial cost

    Parameters:
    props:  list of forward pass output
    lbls:   list of ground truth labeling

    Return:
    err:    cost energy
    cls:    classification error
    grdts:  list of gradient volumes
    """
    grdts = list()
    cls = 0
    err = 0
    for prop, lbl in zip( props, lbls ):
        cls = cls + float(np.count_nonzero( (prop>0.5)!= lbl ))
        grdt = prop.astype('float32') - lbl.astype('float32')
        err = err + np.sum(  -lbl*np.log(prop) - (1-lbl)*np.log(1-prop) )
        grdts.append( grdt )
    cls = cls / float( len(props) )
    err = err / float( len(props) )
    return (err, cls, grdts)

def multinomial_cross_entropy(props, lbls):
    """
    compute multinomial cross entropy

    Parameters:
    props:    list of forward pass output
    lbls:     list of ground truth labeling

    Return:
    err:    cost energy
    cls:    classfication error
    grdts:  list of gradient volumes
    """
    grdts = list()
    cls = 0
    err = 0
    for prop, lbl in zip( props, lbls ):
        cls = cls + float(np.count_nonzero( (prop>0.5) != lbl ))
        grdt = prop.astype('float32') - lbl.astype('float32')
        err = err + np.sum( -lbl * np.log(prop) )
        grdts.append( grdt )
    cls = cls / float( len(props) )
    err = err / float( len(props) )
    return (err, cls, grdts)

def rebalance(grdts, lbls):
    """
    get rebalance weight of gradient.
    make the nonboundary and boundary region have same contribution of training.

    Parameters:
    grdts: list of gradient volumes
    lbls: list of ground truth label
    Return:
    ret: list of balanced gradient volumes
    """
    ret = list()
    for grdt, lbl in zip(grdts,lbls):
        # number of nonzero elements
        num_nz = float( np.count_nonzero(lbl) )
        # total number of elements
        num = float( np.size(lbl) )

        if num_nz == num or num_nz==0:
            ret.append(grdt)
            continue
        # weight of non-boundary and boundary
        wnb = 0.5 * num / num_nz
        wb  = 0.5 * num / (num - num_nz)

        # give value
        weight = np.copy( lbl ).astype('float32')
        weight[lbl>0] = wnb
        weight[lbl==0]= wb
        ret.append( grdt * weight )
    return ret

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

def union_find(ind1, ind2, seg, weight):
    """
    union-find algorithm: weighted quick union with path compression

    Parameters
    ----------
    ind1,ind2:  index of two nodes
    seg:   the segmenation volume with segment id. this array should be flatterned.
    weight: the size of tree

    Return
    ------
    seg:       updated segmentation
    weight:    updated weight
    """
    # find roots
    r1, seg = find_root(ind1, seg)
    r2, seg = find_root(ind2, seg)
    # merge small tree to big tree according to size
    if weight[r1-1] < weight[r2-1]:
        seg[r1-1] = r2
        weight[r2-1] = weight[r2-1] + weight[r1-1]
    return (seg, weight)

def seg_aff( affs, threshold=0.5 ):
    """
    get segmentation from affinity graph using union-find algorithm.
    weighted quick union with path compression: https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf

    Parameters:
    -----------
    affs:  list of affinity graph

    Returns:
    --------
    seg:   segmentation of affinity graph
    """
    # get affinity graphs, copy the array to avoid changing of raw affinity graph
    xaff = np.copy( affs.pop() )
    yaff = np.copy( affs.pop() )
    zaff = np.copy( affs.pop() )
    # remove the boundary edges
    xaff[:,:,0] = 0
    yaff[:,0,:] = 0
    zaff[0,:,:] = 0
    # get edges
    xedges = np.argwhere( xaff>threshold )
    yedges = np.argwhere( yaff>threshold )
    zedges = np.argwhere( zaff>threshold )

    # initialize segmentation with individual label of each voxel
    N = xaff.size
    indmat = np.arange(1, N+1).reshape( xaff.shape )
    seg = np.copy( indmat ).flatten()
    weight = np.ones( seg.shape ).flatten()
    # create edge pair
    for e in xedges:
        # get the index of connected nodes
        ind1 = indmat[e[0], e[1], e[2]]
        ind2 = indmat[e[0], e[1], e[2]-1]
        # union-find algorithm
        seg, weight = union_find(ind1, ind2, seg, weight)
    for e in yedges:
        # get the index of connected nodes
        ind1 = indmat[e[0], e[1],   e[2]]
        ind2 = indmat[e[0], e[1]-1, e[2]]
        # union-find algorithm
        seg, weight = union_find(ind1, ind2, seg, weight)
    for e in zedges:
        # get the index of connected nodes
        ind1 = indmat[e[0]  , e[1], e[2]]
        ind2 = indmat[e[0]-1, e[1], e[2]]
        # union-find algorithm
        seg, weight = union_find(ind1, ind2, seg, weight)

    # relabel all the trees to root id
    it = np.nditer(seg, flags=['f_index'])
    while not it.finished:
        root_ind, seg = find_root(it[0], seg)
        seg[it.index-1] = root_ind

    return seg
def malis(affs, true_affs, masks):
    """
    compute malis weight

    Parameters:
    -----------
    affs:      list of forward pass output affinity graphs, size: Z*Y*X
    true_affs: list of ground truth affinity graphs
    masks:     list of masks of affinity graphs

    Return:
    ------
    err:     cost energy
    cls:     classification error
    grdts:   gradient volumes of affinity graph
    weights: the weight volumes of each affinity edge
    """
    # get segmentation of affinity graph
    seg = seg_aff(affs)
    # sort all the edges
