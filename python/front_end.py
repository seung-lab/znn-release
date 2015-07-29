#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np

def get_sample( vol, insz, lbl, outsz ):
    half_in_sz  = insz.astype('uint32')  / 2
    half_out_sz = outsz.astype('uint32') / 2
    # margin consideration for even-sized input
    margin_sz = half_in_sz - (insz%2)
    set_sz = vol.shape - margin_sz - half_in_sz
    # get random location
    loc = np.zeros(3)
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
    return (vol_in, lbl_out)