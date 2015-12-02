#!/usr/bin/env python
__doc__ = """

Front-End Interface for ZNNv4

Jingpeng Wu <jingpeng.wu@gmail.com>,
Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""

import numpy as np
from emirt import volume_util

def inter_show(start, lc, eta, vol_ins, props, lbl_outs, grdts, pars):
    '''
    Plots a display of training information to the screen
    '''
    import matplotlib.pylab as plt
    name_in, vol  = vol_ins.popitem()
    name_p,  prop = props.popitem()
    name_l,  lbl  = lbl_outs.popitem()
    name_g,  grdt = grdts.popitem()

    m_input = volume_util.crop(vol[0,:,:,:], prop.shape[-3:]) #good enough for now

    # real time visualization
    plt.subplot(251),   plt.imshow(vol[0,0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('input')
    plt.subplot(252),   plt.imshow(m_input[0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('matched input')
    plt.subplot(253),   plt.imshow(prop[0,0,:,:],   interpolation='nearest', cmap='gray')
    plt.xlabel('output')
    plt.subplot(254),   plt.imshow(lbl[0,0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('label')
    plt.subplot(255),   plt.imshow(grdt[0,0,:,:],   interpolation='nearest', cmap='gray')
    plt.xlabel('gradient')

    plt.subplot(256)
    plt.plot(lc.tn_it, lc.tn_err, 'b', label='train')
    plt.plot(lc.tt_it, lc.tt_err, 'r', label='test')
    plt.xlabel('iteration'), plt.ylabel('cost energy')
    plt.subplot(257)
    plt.plot( lc.tn_it, lc.tn_cls, 'b', lc.tt_it, lc.tt_cls, 'r')
    plt.xlabel('iteration'), plt.ylabel( 'classification error' )
    return
