#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np

def square_loss(prop, lbl):
    """
    compute square loss 
    """
    cls = float(np.count_nonzero( (prop>0.5)!= lbl ))
    
    grdt = prop.copy()
    grdt = grdt.astype('float32') - lbl.astype('float32')
    err = np.sum( grdt * grdt ).astype('float32')
    grdt = grdt * 2
    return (err, cls, grdt)