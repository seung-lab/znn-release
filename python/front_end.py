#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import emirt
# numba accelaration
from numba import jit
import time
import matplotlib.pylab as plt

def read_tifs(ftrns, flbls):
    """
    read a list of tif files of original volume and lable
    
    Parameters
    ----------
    ftrns:  list of file name of train volumes
    flbls:  list of file name of lable volumes
    
    Return
    ------
    vols:  list of training volumes
    lbls:  list of labeling volumes
    """
    assert ( len(ftrns) == len(flbls) )
    vols = list()
    lbls = list()
    for ftrn, flbl in zip( ftrns, flbls ):
        vol = emirt.io.imread(ftrn).astype('float32')
        lbl = emirt.io.imread(flbl).astype('float32')
        # normalize the training volume
        vol = vol / 255
        vols.append( vol )
        lbls.append( lbl )
    return (vols, lbls)

def get_sample( vols, insz, lbls, outsz, dp_type='volume' ):
    """
    get random sample from training and labeling volumes
    
    Parameters
    ----------
    vols :  list of training volumes.
    insz :  input size.
    lbls :  list of labeling volumes.
    outsz:  output size of network.
    type :  output data type: volume or affinity graph.
    
    Returns
    -------
    vol_ins  : input volume of network.
    vol_outs : label volume of network.
    """
    # make sure that the input volumes are paired
    assert (len(vols) == len(lbls))
    # pick random volume from volume list
    vid = np.random.randint( len(vols) )
    vol = vols[vid]
    lbl = lbls[vid]
    # configure size    
    half_in_sz  = insz.astype('uint32')  / 2
    half_out_sz = outsz.astype('uint32') / 2
    # margin consideration for even-sized input
    margin_sz = half_in_sz - (insz%2)
    set_sz = vol.shape - margin_sz - half_in_sz
    # get random location
    loc = np.zeros(3)
    

    if 'vol' in dp_type:
        # list of ground truth labels
        vol_ins = np.empty(np.hstack((1,insz)), dtype='float32')
        lbl_outs= np.empty(np.hstack((3,outsz)), dtype='float32')
        loc[0] = np.random.randint(half_in_sz[0], half_in_sz[0] + set_sz[0]-1)
        loc[1] = np.random.randint(half_in_sz[1], half_in_sz[1] + set_sz[1]-1)
        loc[2] = np.random.randint(half_in_sz[2], half_in_sz[2] + set_sz[2]-1)
        # extract volume
        vol_ins[0,:,:,:]  = vol[  loc[0]-half_in_sz[0]  : loc[0]-half_in_sz[0] + insz[0],\
                            loc[1]-half_in_sz[1]  : loc[1]-half_in_sz[1] + insz[1],\
                            loc[2]-half_in_sz[2]  : loc[2]-half_in_sz[2] + insz[2]]
        lbl_outs[0,:,:,:] = lbl[  loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
                            loc[1]-half_out_sz[1] : loc[1]-half_out_sz[1]+outsz[1],\
                            loc[2]-half_out_sz[2] : loc[2]-half_out_sz[2]+outsz[2]]
        lbl_outs = lbl_outs.astype('float32')
    elif 'aff' in dp_type:
        # list of ground truth labels
        vol_ins = np.empty(np.hstack((1,insz)), dtype='float32')
        lbl_outs= np.empty(np.hstack((3,outsz)), dtype='float32')
        
        loc[0] = np.random.randint(half_in_sz[0]+1, half_in_sz[0] + set_sz[0]-1)
        loc[1] = np.random.randint(half_in_sz[1]+1, half_in_sz[1] + set_sz[1]-1)
        loc[2] = np.random.randint(half_in_sz[2]+1, half_in_sz[2] + set_sz[2]-1)
        # extract volume
        vol_ins[0,:,:,:]  = vol[  loc[0]-half_in_sz[0]    : loc[0]-half_in_sz[0] + insz[0],\
                            loc[1]-half_in_sz[1]    : loc[1]-half_in_sz[1] + insz[1],\
                            loc[2]-half_in_sz[2]    : loc[2]-half_in_sz[2] + insz[2]]
        lbl_out = lbl[  loc[0]-half_out_sz[0]-1 : loc[0]-half_out_sz[0]+outsz[0],\
                        loc[1]-half_out_sz[1]-1 : loc[1]-half_out_sz[1]+outsz[1],\
                        loc[2]-half_out_sz[2]-1 : loc[2]-half_out_sz[2]+outsz[2]]
        # z,y,x direction of affinity
        lbl_outs[0,:,:,:] = (lbl_out[1:,1:,1:] == lbl_out[:-1,1:,1:]) & (lbl_out[1:,1:,1:]>0)
        lbl_outs[1,:,:,:] = (lbl_out[1:,1:,1:] == lbl_out[1:,:-1,1:]) & (lbl_out[1:,1:,1:]>0)
        lbl_outs[2,:,:,:] = (lbl_out[1:,1:,1:] == lbl_out[1:,1:,:-1]) & (lbl_out[1:,1:,1:]>0)
        lbl_outs = lbl_outs.astype('float32')
    else:
        raise NameError('unknown mode type.')

    return (vol_ins, lbl_outs)

def data_norm( data ):
    """
    normalize data for network
    centerize to zero; adjust range to [-1,1]
    
    Parameters
    ----------
    data : 4D array
    
    Returns
    -------
    data : 4D array
    """
    # centerize to zero    
    data = data - np.mean(data)
    # adjust range to [-1,1]
#    np.    
    
@jit(nopython=True)
def data_aug_transform(data, rft):
    """
    transform data according to a rule
    
    Parameters
    ----------
    data : 3D numpy array need to be transformed
    rft : transform rule
    
    Returns
    -------
    data : the transformed array
    """
    # transform every pair of input and label volume
    if rft[0]:
        # first flip and than transpose
        if rft[1]:
            data  = np.fliplr( data )
            if rft[2]:
                data  = np.flipud( data )
                if rft[3]:
                    data = data[::-1, :,:]
        if rft[4]:
            data = data.transpose(0,2,1)
    else:
        # first transpose, than flip
        if rft[4]:
            data = data.transpose(0,2,1)
        if rft[1]:
            data = np.fliplr( data )
            if rft[2]:
                data = np.flipud( data )
                if rft[3]:
                    data = data[::-1, :,:]
    return data

#@jit(nopython=True)
def data_aug( vols, lbls ):
    """
    data augmentation, transform volumes randomly to enrich the training dataset.
    
    Parameters
    ----------
    vol : input volumes of network.
    lbl : label volumes of network.
    
    Returns
    -------
    vol : transformed input volumes of network.
    lbl : transformed label volumes.
    """
    # random flip and transpose: flip-transpose order, fliplr, flipud, flipz, transposeXY
    rft = (np.random.random(5)>0.5)
    for i in xrange(vols.shape[0]):
        vols[i,:,:,:] = data_aug_transform(vols[i,:,:,:], rft)
    for i in xrange(lbls.shape[0]):
        lbls[i,:,:,:] = data_aug_transform(lbls[i,:,:,:], rft)
    return (vols, lbls)

def inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                eta, vol_ins, props, lbl_outs, grdts ):
    # time
    elapsed = time.time() - start
    print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s, learning rate: %.4f"\
            %(i, err, cls, elapsed, eta )
    # real time visualization
    plt.subplot(331),   plt.imshow(vol_ins[0,0,:,:],       interpolation='nearest', cmap='gray')
    plt.xlabel('input')
    plt.subplot(332),   plt.imshow(props[1,0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('inference')
    plt.subplot(333),   plt.imshow(lbl_outs[1,0,:,:], interpolation='nearest', cmap='gray')
    plt.xlabel('lable')
    plt.subplot(334),   plt.imshow(grdts[1,0,:,:],     interpolation='nearest', cmap='gray')
    plt.xlabel('gradient')

    
    plt.subplot(337), plt.plot(it_list, err_list, 'r')
    plt.xlabel('iteration'), plt.ylabel('cost energy')
    plt.subplot(338), plt.plot(it_list, cls_list, 'b')
    plt.xlabel('iteration'), plt.ylabel( 'classification error' )
        
    plt.pause(1)

    # reset time
    start = time.time()
    # reset err and cls
    err = 0
    cls = 0
    return start