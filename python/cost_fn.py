#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np

def square_loss(prop, lbl):
    """
    compute square loss 
    
    Parameters:
    prop:   forward pass output
    lbl:    ground truth labeling
    """
    cls = float(np.count_nonzero( (prop>0.5)!= lbl ))
    
    grdt = prop.copy()
    grdt = grdt.astype('float32') - lbl.astype('float32')
    err = np.sum( grdt * grdt ).astype('float32')
    grdt = grdt * 2
    return (err, cls, grdt)

def rebalance(lbls):
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
