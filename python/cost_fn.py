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

def rebalance(lbls):
    """
    get rebalance weight of gradient.
    make the nonboundary and boundary region have same contribution of training.
    
    Parameters:
    lbls: list of ground truth label
    Return:
    weights: list of weight of gradient
    """
    weights = list()
    for lbl in lbls:
        # number of nonzero elements
        num_nz = float( np.count_nonzero(lbl) )
        # total number of elements
        num = float( np.size(lbl) )
        # weight of non-boundary and boundary
        wnb = 0.5 * num / num_nz
        wb  = 0.5 * num / (num - num_nz)
        
        # give value
        weight = np.copy( lbl ).astype('float32')
        weight[lbl>0] = wnb
        weight[lbl==0]= wb
        weights.append( weight )
    return weights
    

def malis_weight(affs, true_affs, masks):
    """
    compute malis weight 
    
    Parameters:
    affs:   forward pass output affinity graphs, size: 3*Z*Y*X
    true_affs: ground truth affinity graphs 
    masks:  masks of affinity graphs
    
    Return:
    weight: the weight of each affinity edge
    """
