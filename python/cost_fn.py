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
    get rebalance tree_size of gradient.
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
        # tree_size of non-boundary and boundary
        wnb = 0.5 * num / num_nz
        wb  = 0.5 * num / (num - num_nz)

        # give value
        tree_size = np.copy( lbl ).astype('float32')
        tree_size[lbl>0] = wnb
        tree_size[lbl==0]= wb
        ret.append( grdt * tree_size )
    return ret

def malis(affs, true_affs, threshold=0.5):
    """
    compute malis tree_size

    Parameters:
    -----------
    affs:      list of forward pass output affinity graphs, size: Z*Y*X
    true_affs: list of ground truth affinity graphs


    Return:
    ------
    err:     cost energy
    cls:     classification error
    grdts:   gradient volumes of affinity graph
    tree_sizes: the tree_size volumes of each affinity edge
    """
    # get affinity graphs
    xaff = affs.pop()
    yaff = affs.pop()
    zaff = affs.pop()

    # initialize segmentation with individual label of each voxel
    N = xaff.size
    ids = np.arange(1, N+1).reshape( xaff.shape )
    seg = np.copy( ids ).flatten()
    tree_size = np.ones( seg.shape ).flatten()

    # initialize edges
    edges = list()

    for z in xrange(seg.shape[0]):
        for y in xrange(seg.shape[1]):
            for x in xrange(1,seg.shape[2]):
                if xaff[z,y,x] > threshold:
                    edges.append( xaff[z,y,x], ids[z,y,x], ids[z,y,x-1] )
    for z in xrange(seg.shape[0]):
        for y in xrange(1,seg.shape[1]):
            for x in xrange(seg.shape[2]):
                if yaff[z,y,x]>threshold:
                    edges.append( yaff[z,y,x], ids[z,y,x], ids[z,y-1,x] )
    for z in xrange(1,seg.shape[0]):
        for y in xrange(seg.shape[1]):
            for x in xrange(seg.shape[2]):
                if zaff[z,y,x]>threshold:
                    edges.append( zaff[z,y,x], ids[z,y,x], ids[z-1,y,x] )
    # descending sort
    edges.sort(reverse=True)

    # find the maximum-spanning tree based on union-find algorithm
    import emirt
    for e in edges:
        # find operation with path compression
        r1,seg = emirt.volume_util.find_root(e[1], seg)
        r2,seg = emirt.volume_util.find_root(e[2], seg)
        
        if r1!=r2:
            # not in a same set, this a maximin edge
            # merge or union the two sets or trees
            seg, tree_size = emirt.volume_util.union_tree(r1, r2, seg, tree_size)

def mask_filter(grdts, masks):
    """
    eliminate some region of gradient using a mask

    Parameters
    ----------
    grdts:  list of gradient volumes
    masks:  list of masks of affinity graphs

    Return
    ------
    grdts:  list of gradient volumes
    """
    ret = list()
    for grdt, mask in zip(grdts, masks):
        ret.append( grdt * mask )
    return ret
