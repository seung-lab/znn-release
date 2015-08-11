#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import emirt
# numba accelaration
from numba.decorators import autojit

@autojit
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

#@autojit
def get_sample( vols, insz, lbls, outsz, type='volume' ):
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
    # list of ground truth labels
    lbls = list()

    if 'vol' in type:
        loc[0] = np.random.randint(half_in_sz[0], half_in_sz[0] + set_sz[0]-1)
        loc[1] = np.random.randint(half_in_sz[1], half_in_sz[1] + set_sz[1]-1)
        loc[2] = np.random.randint(half_in_sz[2], half_in_sz[2] + set_sz[2]-1)
        # extract volume
        vol_in  = vol[  loc[0]-half_in_sz[0]  : loc[0]-half_in_sz[0] + insz[0],\
                    loc[1]-half_in_sz[1]  : loc[1]-half_in_sz[1] + insz[1],\
                    loc[2]-half_in_sz[2]  : loc[2]-half_in_sz[2] + insz[2]]
        lbl_out = lbl[  loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
                        loc[1]-half_out_sz[1] : loc[1]-half_out_sz[1]+outsz[1],\
                        loc[2]-half_out_sz[2] : loc[2]-half_out_sz[2]+outsz[2]]
        lbls.append( (lbl_out>0).astype('float32') )
    elif 'aff' in type:
        loc[0] = np.random.randint(half_in_sz[0]+1, half_in_sz[0] + set_sz[0]-1)
        loc[1] = np.random.randint(half_in_sz[1]+1, half_in_sz[1] + set_sz[1]-1)
        loc[2] = np.random.randint(half_in_sz[2]+1, half_in_sz[2] + set_sz[2]-1)
        # extract volume
        vol_in  = vol[  loc[0]-half_in_sz[0]    : loc[0]-half_in_sz[0] + insz[0],\
                        loc[1]-half_in_sz[1]    : loc[1]-half_in_sz[1] + insz[1],\
                        loc[2]-half_in_sz[2]    : loc[2]-half_in_sz[2] + insz[2]]
        lbl_out = lbl[  loc[0]-half_out_sz[0]-1 : loc[0]-half_out_sz[0]+outsz[0],\
                        loc[1]-half_out_sz[1]-1 : loc[1]-half_out_sz[1]+outsz[1],\
                        loc[2]-half_out_sz[2]-1 : loc[2]-half_out_sz[2]+outsz[2]]
        # z,y,x direction of affinity
        lbl_z = (lbl_out[1:,1:,1:] == lbl_out[:-1,1:,1:]) & (lbl_out[1:,1:,1:]>0)
        lbl_y = (lbl_out[1:,1:,1:] == lbl_out[1:,:-1,1:]) & (lbl_out[1:,1:,1:]>0)
        lbl_x = (lbl_out[1:,1:,1:] == lbl_out[1:,1:,:-1]) & (lbl_out[1:,1:,1:]>0)
        lbls.append( lbl_z.astype('float32') )
        lbls.append( lbl_y.astype('float32') )
        lbls.append( lbl_x.astype('float32') )
    else:
        raise NameError('unknown mode type.')

    vol_ins = list()
    vol_ins.append( np.ascontiguousarray( vol_in ) )
    return (vol_ins, lbls)

@autojit
def data_aug( vol_ins, lbl_outs ):
    """
    data augmentation, transform volumes randomly to enrich the training dataset.
    
    Parameters
    ----------
    vol_ins  :  input volumes of network.
    lbl_outs : label volumes of network.
    
    Returns
    -------
    vol_ins2  : transformed input volumes of network.
    vol_outs2 : transformed label volumes.
    """
    vol_ins2 = list()
    vol_outs2 = list()
    # random flip and transpose: flip-transpose order, fliplr, flipud, flipz, transposeXY
    rft = (np.random.random(5)>0.5)
    for vin, vout in zip(vol_ins, lbl_outs):
        # transform every pair of input and label volume
        if rft[0]:
            # first flip and than transpose
            if rft[1]:
                vin  = np.fliplr( vin )
                vout = np.fliplr( vout )
                if rft[2]:
                    vin  = np.flipud( vin )
                    vout = np.flipud( vout )
                    if rft[3]:
                        vin =   vin[::-1, :,:]
                        vout = vout[::-1, :,:]
            vin = vin.transpose(0,2,1)
        else:
            # first transpose, than flip
            vin = vin.transpose(0,2,1)
            if rft[1]:
                vin  = np.fliplr( vin )
                vout = np.fliplr( vout )
                if rft[2]:
                    vin  = np.flipud( vin )
                    vout = np.flipud( vout )
                    if rft[3]:
                        vin =   vin[::-1, :,:]
                        vout = vout[::-1, :,:]
        vol_ins2.append( vin )
        vol_outs2.append( vout )
    return (vol_ins2, vol_outs2)
    