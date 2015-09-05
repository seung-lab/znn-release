#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import utils
import cost_fn

def _single_test(net, pars, sample, insz, outsz):
    vol_ins, lbl_outs = sample.get_random_sample( insz, outsz )
   
    # forward pass
    props = net.forward( utils.loa_as_continue(vol_ins, dtype='float32') )
   
    # cost function and accumulate errors
    props, err, grdts = pars['cost_fn']( props, lbl_outs )
    cls = cost_fn.get_cls(props, lbl_outs)
    
    # normalize
    err = err / utils.loa_vox_num(props)
    cls = cls / utils.loa_vox_num(props)
    return err, cls

def znn_test(net, pars, samples, insz, outsz, terr_list, tcls_list):
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
    err = 0
    cls = 0
    net.set_phase(1)
    test_num = pars['test_num']
    for i in xrange( test_num ):
        cerr, ccls = _single_test(net, pars, samples, insz, outsz)
        err = err + cerr
        cls = cls + ccls
    net.set_phase(0)
    
    terr_list.append( err/test_num )
    tcls_list.append( cls/test_num )
    
    return terr_list, tcls_list