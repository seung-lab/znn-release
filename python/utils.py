#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import numpy as np

def make_continuous( d , dtype='float32'):
    """
    make the dictionary arrays continuous.
    
    Parameters
    ----------
    d : dict, the input dictionary of 4D array.
    
    Returns
    -------
    d : dict, the inner array are continuous.    
    """
    for name, arr in d.iteritems():
        d[name] = np.ascontiguousarray(arr, dtype=dtype)
    return d

def get_vox_num( d ):
    n = 0
    for name, arr in d.iteritems():
        n = n + arr.shape[0]*arr.shape[1]*arr.shape[2]*arr.shape[3]
    return n
def get_total_num(outputs):
    """
    """
    n = 0
    for name, sz in outputs.iteritems():
        n = n + np.prod(sz)
    return n

def save_statistics( pars, it_list, err_list, cls_list,\
                        titr_list, terr_list, tcls_list):
    # get filename
    fname = pars['train_save_net']
    import os
    root, ext = os.path.splitext(fname)
    fname = root + '_statistics_current.h5'
    if os.path.exists( fname ):
        os.remove( fname )
    
    # save variables
    import h5py
    f = h5py.File( fname )    
    f.create_dataset('train/it',  data=it_list)
    f.create_dataset('train/err', data=err_list)
    f.create_dataset('train/cls', data=cls_list)
    f.create_dataset('test/it',   data=titr_list)
    f.create_dataset('test/err',  data=terr_list)
    f.create_dataset('test/cls',  data=tcls_list)
    f.close()
    
    # move to new name
    fname2 = root + '_statistics.h5'
    os.rename(fname, fname2)
    