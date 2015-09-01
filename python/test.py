#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import cost_fn
import front_end

def znn_test(net, pars, sample, insz, outsz, terr_list, tcls_list):
    """
    test the net 
    
    Parameters
    ----------
    net : network
    pars : dict, parameters
    sample : a input and output sample
    insz : 1D array, input size
    outsz : 1D array, output size
    terr_list : list of float32, test cost
    tcls_list : list of float32, test classification error
    
    Returns
    -------
    terr_list : list of float32, test cost
    tcls_list : list of float32, test classification error
    """
    vol_ins, lbl_outs = sample.get_random_sample( insz, outsz )
   
    # forward pass
    props = net.forward( np.ascontiguousarray(vol_ins) ).astype('float32')

    # softmax
    if pars['cost_fn_str']=='multinomial_cross_entropy':
        props = cost_fn.softmax(props)
    
    # cost function and accumulate errors
    err, grdts = pars['cost_fn']( props, lbl_outs )
    cls = np.count_nonzero( (props>0.5)!= lbl_outs )
    
    # normalize
    err = err / float( props.shape[0] * outsz[0] * outsz[1] * outsz[2])
    cls = cls / float( props.shape[0] * outsz[0] * outsz[1] * outsz[2])
    
    terr_list.append( err )
    tcls_list.append( cls )
    return terr_list, tcls_list