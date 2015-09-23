#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
# numba accelaration
#from numba import jit

def get_cls(props, lbls):
    """
    compute classification error.
    
    Parameters
    ----------
    props : dict of array, network propagation output volumes.
    lbls  : dict of array, ground truth 
    
    Returns
    -------
    c : number of classification error
    """
    c = 0.0
    for name, prop in props.iteritems():
        lbl = lbls[name]            
        c = c + np.count_nonzero( (prop>0.5)!= lbl )
        
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
    grdts = dict()
    err = 0
    for name, prop in props.iteritems():
        lbl = lbls[name]
        grdt = prop - lbl
        # cost and classification error
        err = err + np.sum( grdt * grdt )
        grdts[name] = grdt * 2
        
        print "gradient: ", grdts[name]        
        
    return (props, err, grdts)

#@jit(nopython=True)
def binomial_cross_entropy(props, lbls):
    """
    compute binomial cost

    Parameters
    ----------
    props:  dict of network output arrays
    lbls:   dict of ground truth arrays

    Return
    ------
    err:    cost energy
    grdts:  dict of gradient volumes
    """
    grdts = dict()
    err = 0
    for name, prop in props.iteritems():
        lbl = lbls[name]
        grdts[name] = prop - lbl
        err = err + np.nansum(  -lbl*np.log(prop) - (1-lbl)*np.log(1-prop) )
    return (props, err, grdts)

#@jit(nopython=True)
def softmax(props):
    """
    softmax activation

    Parameters:
    props:  numpy array, net forward output volumes

    Returns:
    ret:   numpy array, softmax activation volumes
    """
    ret = dict()
    for name, prop in props.iteritems():
        # make sure that it is the output of binary class
        assert(prop.shape[0]==2)

        # rebase the prop for numerical stabiligy
        # mathimatically, this do not affect the softmax result!
        # http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
#        prop = prop - np.max(prop)
        propmax = np.max(prop, axis=0)
        prop[0,:,:,:] -= propmax
        prop[1,:,:,:] -= propmax  
        
#        log_softmax = np.empty(prop.shape, dtype='float32')
#        log_softmax[0,:,:,:] = prop[0,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
#        log_softmax[1,:,:,:] = prop[1,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
#        
#        ret[name] = np.exp( log_softmax )
        
        prop = np.exp(prop)
        pesum = np.sum(prop, axis=0)
        ret[name] = np.empty(prop.shape, dtype=prop.dtype)
        for c in xrange(prop.shape[0]):
            ret[name][c,:,:,:] = prop[c,:,:,:] / pesum
        
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
    grdts = dict()
    err = 0
    for name, prop in props.iteritems():
        lbl = lbls[name]
        grdts[name] = prop - lbl
        err = err + np.sum( -lbl * np.log(prop) )
    return (props, err, grdts)

def softmax_loss(props, lbls):
#    for name, prop in props.iteritems():
#        print "prop before softmax: ", prop
#        assert(not np.any(np.isnan(prop)))
        
    props = softmax(props)
    
#    for name, prop in props.iteritems():
#        print "prop after softmax: ", prop
#        assert(not np.any(np.isnan(prop)))
    return multinomial_cross_entropy(props, lbls)
    
def softmax_loss2(props, lbls):
    grdts = dict()
    err = 0
    
    for name, prop in props.iteritems():
        # make sure that it is the output of binary class
        assert(prop.shape[0]==2)
        
        print "original prop: ", prop        
        
        # rebase the prop for numerical stabiligy
        # mathimatically, this do not affect the softmax result!
        # http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
#        prop = prop - np.max(prop)
        propmax = np.max(prop, axis=0)
        prop[0,:,:,:] -= propmax
        prop[1,:,:,:] -= propmax        
        
        log_softmax = np.empty(prop.shape, dtype=prop.dtype)
        log_softmax[0,:,:,:] = prop[0,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
        log_softmax[1,:,:,:] = prop[1,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
        prop = np.exp(log_softmax)
        props[name] = prop

        lbl = lbls[name]
        grdts[name] = prop - lbl
        err = err + np.sum( -lbl * log_softmax )
        print "gradient: ", grdts[name]
        assert(not np.any(np.isnan(grdts[name])))
    return (props, err, grdts)

#def hinge_loss(props, lbls):
# TO-DO

#@jit(nopython=True)
# TO-DO, not fully implemented
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
    import emirt
#    seg = emirt.volume_util.seg_aff(true_affs)
    if isinstance(affs, dict):
        assert( len(affs.keys())==1 )
        key = affs.keys()[0]
        affs = affs.values()[0]
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
    weights = np.zeros( np.hstack((affs.shape[0], xaff.size)), dtype=affs.dtype )
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
    # transform to dictionary
    ret = dict()
    ret[key] = weights
    return ret

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
