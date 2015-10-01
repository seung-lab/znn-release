#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import numpy as np
from numba import autojit

def data_aug_transform(data, rft):
        """
        transform data according to a rule

        Parameters
        ----------
        data : 3D numpy array need to be transformed
        rft : transform rule, specified as an array of bool
            [z-reflection,
            y-reflection,
            x-reflection,
            xy transpose]

        Returns
        -------
        data : the transformed array
        """

        if np.size(rft)==0:
            return data
        # transform every pair of input and label volume

        #z-reflection
        if rft[0]:
            data  = data[:, ::-1, :,    :]
        #y-reflection
        if rft[1]:
            data  = data[:, :,    ::-1, :]
        #x-reflection
        if rft[2]:
            data = data[:,  :,    :,    ::-1]
        #transpose
        if rft[3]:
            data = data.transpose(0,1,3,2)

        return data

def _mirror2d( im, bf, fov ):
    """
    mirror image in 2D

    Parameters
    ----------
    im : 2D array
    bf : buffer for mirrored image
    fov : 2D vector

    Returns
    -------
    bf : mirrored buffer
    """
    bsz = np.asarray(bf.shape, dtype='int')
    isz = np.asarray(im.shape, dtype='int')
    fov = fov.astype('int32')
    l = (fov-1)/2
    b = bsz - (fov/2)
    i = isz - (fov/2)

    # 4 edges
    bf[:l[0], l[1]:b[1]] = im[:l[0], :][::-1, :]
    bf[l[0]:b[0], :l[1]] = im[:, :l[1]][:, ::-1]

    bf[b[0]:, l[1]:b[1]] = im[i[0]:, :][::-1, :]
    bf[l[0]:b[0], b[1]:] = im[:, i[1]:][:, ::-1]

    # 4 corners
    bf[:l[0], :l[1]] = im[:l[0], :l[1]][::-1,::-1]
    bf[b[0]:, b[1]:] = im[i[0]:, i[1]:][::-1,::-1]
    bf[:l[0], b[1]:] = im[:l[0], i[1]:][::-1,::-1]
    bf[b[0]:, :l[1]] = im[i[0]:, :l[1]][::-1,::-1]
    return bf

def boundary_mirror( arr, fov ):
    """
    mirror the boundary for each 3d array

    Parameters
    ----------
    arr : 4D array.
    fov : vector with 3 int number, field of view.

    Return
    ------
    ret : expanded 4D array with mirrored boundary
    """
    assert(np.size(fov)==3)
    print "boundary mirror..."
    fov = fov.astype('int32')
    if np.all(fov==1):
        return arr
    # buffer size
    bfsz = np.asarray(arr.shape, dtype='int32')
    bfsz[1:] += fov-1
    # initialize the buffer
    bf = np.zeros(tuple(bfsz), dtype=arr.dtype)

    # low and high of fov
    l = (fov-1)/2
    b = bfsz[1:] - fov/2
    # fill the buffer with existing array
    bf[:, l[0]:b[0], l[1]:b[1], l[2]:b[2]] = arr
    for c in xrange(arr.shape[0]):
        for z in xrange(arr.shape[1]):
            bf[c,z+l[0],:,:] = _mirror2d(arr[c, z, :, :], bf[c,z+l[0],:,:], fov[1:])
        for y in xrange(arr.shape[2]):
            bf[c,:,y+l[1],:] = _mirror2d(arr[c, :, y, :], bf[c,:,y+l[1],:], fov[0:3:2])
        for x in xrange(arr.shape[3]):
            bf[c,:,:,x+l[2]] = _mirror2d(arr[c, :, :, x], bf[c,:,:,x+l[2]], fov[:2])

        # repeat mirroring z sections for filling 8 corners
        for z in xrange(l[0]):
            bf[c,z,:,:] = _mirror2d(bf[c, z, l[1]:b[1], l[2]:b[2]], bf[c,z,:,:], fov[1:])
        for z in xrange(b[0],bfsz[1]):
            bf[c,z,:,:] = _mirror2d(bf[c, z, l[1]:b[1], l[2]:b[2]], bf[c,z,:,:], fov[1:])
    return bf

@autojit(nopython=True)
def fill_boundary( lbl ):
    """
    separate the contacting segments with boundaries.
    """
    assert(len(lbl.shape)==3)
    for z in xrange( lbl.shape[0] ):
        for y in xrange( lbl.shape[1]-1 ):
            for x in xrange( lbl.shape[2]-1 ):
                if lbl[z,y,x]>0:
                    if lbl[z,y,x]!=lbl[z,y+1,x] and lbl[z,y+1,x]>0:
                        lbl[z,y,x] = 0
                        lbl[z,y+1] = 0
                    if lbl[z,y,x]!=lbl[z,y,x+1] and lbl[z,y,x+1]>0:
                        lbl[z,y,x] = 0
                        lbl[z,y,x+1] = 0
    return lbl

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

def dict_mul(das,dbs):
    ret = dict()
    for name, a in das.iteritems():
        b = dbs[name]
        if np.size(b)==np.size(a):
            ret[name] = a * b
        elif np.size(b)==0:
            ret[name] = a
    return ret

def save_statistics( pars, it_list, err_list, cls_list,\
                        titr_list, terr_list, tcls_list, elapsed):
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
    f.create_dataset('elapsed',   data=elapsed)
    f.close()

    # move to new name
    fname2 = root + '_statistics.h5'
    os.rename(fname, fname2)