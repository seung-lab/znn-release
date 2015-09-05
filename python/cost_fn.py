#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import utils
# numba accelaration
#from numba import jit

def get_cls(props, lbls):
    """
    compute classification error.
    
    Parameters
    ----------
    props : list of array, network propagation output volumes.
    lbls  : list of array, ground truth 
    
    Returns
    -------
    c : number of classification error
    """
    c = 0.0
    for prop, lbl in zip(props, lbls):
        c = c + np.count_nonzero( (prop>0.5)!= lbl )
#    c = c / utils.loa_vox_num(props)
    return c
#@jit(nopython=True)
def square_loss(props, lbls):
    """
    compute square loss

    Parameters
    ----------
    props: numpy array, forward pass output
    lbls:  numpy array, ground truth labeling

    Return
    ------
    err:   cost energy
    grdts: numpy array, gradient volumes
    """
    assert(len(props)==len(lbls))
    grdts = list()
    err = 0.0
    for prop, lbl in zip(props, lbls):
        grdt = prop - lbl
        # cost and classification error
        err = err + np.sum( grdt * grdt )
        grdt = grdt * 2
        grdts.append( grdt )
    return (props, err, grdts)

#@jit(nopython=True)
def binomial_cross_entropy(props, lbls):
    """
    compute binomial cost

    Parameters
    ----------
    props:  forward pass output
    lbls:   ground truth labeling

    Return
    ------
    err:    cost energy
    grdts:  list of gradient volumes
    """
#    assert(props.shape==lbls.shape)
    grdts = utils.loa_sub(props, lbls)
    err = 0
    for prop, lbl in zip(props, lbls):
        err = err + np.sum(  -lbl*np.log(prop) - (1-lbl)*np.log(1-prop) )
    return (props, err, grdts)

#@jit(nopython=True)
def softmax(props):
    """
    softmax activation

    Parameters:
    props:  list of numpy array, net forward output volumes

    Returns:
    ret:   list of numpy array, softmax activation volumes
    """
    
    pes = list()
    pesum = np.zeros( props[0].shape, dtype='float32' )
    for prop in props:
        prop_exp = np.exp(prop.astype('float32'))
        pes.append( prop_exp )
        pesum = pesum + prop_exp
        
    ret = list()
    for pe in pes:
        ret.append( pe / pesum )
    return ret

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
    assert(len(props)==len(lbls))
    grdts = list()
    err = 0.0
    for prop, lbl in zip(props, lbls):
        grdts.append( prop - lbl )
        err = err + np.sum( -lbl * np.log(prop) )
    return (props, err, grdts)

#@jit(nopython=True)
def softmax_loss(props, lbls):
    props = softmax(props)
    return multinomial_cross_entropy(props, lbls)

#def hinge_loss(props, lbls):
# TO-DO

#@jit(nopython=True)
def malis_weight(affs, true_affs, threshold=0.5):
    """
    compute malis tree_size

    Parameters:
    -----------
    affs:      4D array of forward pass output affinity graphs, size: C*Z*Y*X
    true_affs : 4d array of ground truth affinity graph
    threshold: threshold for segmentation

    Return:
    ------
    weights : 4D array of weights
    """
#    seg = segment(true_affs)
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
    weights = np.zeros( np.hstack((affs.shape[0], xaff.size)), dtype='float32' )
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
            weights[e[3], r1-1] = weights[e[3],r1-1] + s1*s2
#            print "s1: %d, s2: %d"%(s1,s2)
            # merge the two sets/trees
            seg, tree_size = emirt.volume_util.union_tree(r1, r2, seg, tree_size)
    # normalize the weights
    N = float(N)
    weights = weights * (3*N) / ( N*(N-1)/2 )
    weights = weights.reshape( affs.shape )
    return weights

def sparse_cost(outputs, labels, cost_fn):
    """
    Sparse Versions of Pixel-Wise Cost Functions

    Parameters
    ----------
    outputs: numpy array, forward pass output
    labels:  numpy array, ground truth labeling
    cost_fn: function to make sparse

    Return
    ------
    err:   cost energy
    grdts: numpy array, gradient volumes
    """

    flat_outputs = outputs[labels != 0]
    flat_labels = labels[labels != 0]

    errors, gradients = cost_fn(flat_outputs, flat_labels)

    # full_errors = np.zeros(labels.shape)
    full_gradients = np.zeros(labels.shape)

    # full_errors[labels != 0] = errors
    full_gradients[labels != 0] = gradients

    return (errors, full_gradients)
