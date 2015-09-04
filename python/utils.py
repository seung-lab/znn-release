#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import numpy as np
import emirt

def loa_add(arrs1, arrs2):
    ret = list()
    for arr1, arr2 in zip( arrs1, arrs2 ):
        ret.append( arr1 + arr2 )
    return ret

def loa_sub(arrs1, arrs2):
    ret = list()
    for arr1, arr2 in zip( arrs1, arrs2 ):
        ret.append( arr1 - arr2 )
    return ret
def loa_mul(arrs1, arrs2):
    ret = list()
    for arr1, arr2 in zip( arrs1, arrs2 ):
        ret.append( arr1 * arr2 )
    return ret
def loa_div(arrs1, arrs2):
    ret = list()
    for arr1, arr2 in zip( arrs1, arrs2 ):
        ret.append( arr1 / arr2 )
    return ret
    
def binarize(arrs1, dtype='float32'):
    ret = list()
    for arr in arrs1:
        ret.append( (arr>0).astype(dtype) )
    return ret


def _center_crop(self, vol, shape):
    """
    crop the volume from the center

    Parameters
    ----------
    vol : the array to be croped
    shape : the croped shape

    Returns
    -------
    vol : the croped volume
    """
    sz1 = np.asarray( vol.shape )
    sz2 = np.asarray( shape )
    # offset of both sides
    off1 = (sz1 - sz2+1)/2
    off2 = (sz1 - sz2)/2
    return vol[ off1[0]:-off2[0],\
                off1[1]:-off2[1],\
                off1[2]:-off2[2]]
def auto_crop(arrs):
    """
    crop the list of volumes to make sure that volume sizes are the same.
    Note that this function was not tested yet!!
    """
    if len(arrs) == 1:
        return arrs
    
    # find minimum size
    splist = list()
    for arr in arrs:
        splist.append( arr.shape )
    sz_min = min( splist )

    # crop every volume
    ret = list()
    for k in xrange( len(arrs) ):
        ret.append( _center_crop( arrs[k], sz_min ) )
    return ret

def _preprocess_vol( vol, pp_type):
    if 'standard2D' == pp_type:
        for z in xrange( vol.shape[0] ):
            vol[z,:,:] = (vol[z,:,:] - np.mean(vol[z,:,:])) / np.std(vol[z,:,:])
    elif 'standard3D' == pp_type:
        vol = (vol - np.mean(vol)) / np.std(vol)
    elif 'none' == pp_type:
        return vol
    else:
        raise NameError( 'invalid preprocessing type' )
    return vol

def preprocess(arrs, pp_types):
    ret = list()
    for vol, pp_type in zip(arrs, pp_types):
        ret.append( _preprocess_vol(vol, pp_type) )
    return ret

def read_files( files):
    """
    read a list of tif files of original volume and lable

    Parameters
    ----------
    files : list of string, file names

    Return
    ------
    ret:  list of 3D array
    """
    ret = list()
    for fl in files:
        vol = emirt.emio.imread(fl).astype('float32')
        ret.append( vol )
    return ret

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
    