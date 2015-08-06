#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np

def square_loss(props, lbls):
    """
    compute square loss

    Parameters
    ----------
    props:   list of forward pass output
    lbls:    list of ground truth labeling
    
    Return
    ------
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

    Parameters
    ----------
    props:  list of forward pass output
    lbls:   list of ground truth labeling

    Return
    ------
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

    Parameters
    ----------
    props:    list of forward pass output
    lbls:     list of ground truth labeling

    Return
    ------
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

#def hinge_loss(props, lbls):
    
def weight_gradient(grdts, weights):
    grdts[:] = [ grdt*weight for grdt, weight in zip(grdts, weights) ]        
    return grdts

def rebalance( lbls ):
    """
    get rebalance tree_size of gradient.
    make the nonboundary and boundary region have same contribution of training.

    Parameters
    ----------
    grdts: list of gradient volumes
    lbls: list of ground truth label
    
    Return
    ------
    ret: list of balanced gradient volumes
    """
    weights = list()
    for lbl in lbls:
        # number of nonzero elements
        num_nz = float( np.count_nonzero(lbl) )
        # total number of elements
        num = float( np.size(lbl) )

        if num_nz == num or num_nz==0:
            weights.append( np.ones(lbl.shape) )
            continue
        # weight of non-boundary and boundary
        wnb = 0.5 * num / num_nz
        wb  = 0.5 * num / (num - num_nz)

        # give value
        weight = np.copy( lbl ).astype('float32')
        weight[lbl>0] = wnb
        weight[lbl==0]= wb
        weights.append( weight )
    return weights

def malis_weights(affs, threshold=0.5):
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
    xaff = affs[2]
    yaff = affs[1]
    zaff = affs[0]
    shape = xaff.shape

    # initialize segmentation with individual label of each voxel
    N = xaff.size
    ids = np.arange(1, N+1).reshape( xaff.shape )
    seg = np.copy( ids ).flatten()
    tree_size = np.ones( seg.shape ).flatten()

    # initialize edges: aff, id1, id2, z/y/x, true_aff
    edges = list()
    weights = list()
    # z,y,x of weight
    weights.append( np.zeros(seg.size) )
    weights.append( np.zeros(seg.size) )
    weights.append( np.zeros(seg.size) )

    for z in xrange(shape[0]):
        for y in xrange(shape[1]):
            for x in xrange(1,shape[2]):
                edges.append( (xaff[z,y,x], ids[z,y,x], ids[z,y,x-1], 2) )
    for z in xrange(shape[0]):
        for y in xrange(1,shape[1]):
            for x in xrange(shape[2]):
                edges.append( (yaff[z,y,x], ids[z,y,x], ids[z,y-1,x], 1) )
    for z in xrange(1,shape[0]):
        for y in xrange(shape[1]):
            for x in xrange(shape[2]):
                edges.append( (zaff[z,y,x], ids[z,y,x], ids[z-1,y,x], 0) )
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
            # get the size of two sets/trees 
            s1 = tree_size[r1-1]
            s2 = tree_size[r2-1]
            # accumulate weights
            weights[e[3]][r1-1] = weights[e[3]][r1-1] + s1*s2
            # merge the two sets/trees
            seg, tree_size = emirt.volume_util.union_tree(r1, r2, seg, tree_size)
    # normalize the weights
    weights[:] = [weight.astype('float32') * (3*N) / ( N*(N-1)/2 ) for weight in weights]
    weights[:] = [weight.reshape( shape ) for weight in weights]
    return weights
   
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
