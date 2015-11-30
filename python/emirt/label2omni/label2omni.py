# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:45:44 2015

@author: jingpeng
"""
import h5py
import emirt
import numpy as np
import os

#%% raw data
def chann(infname, outfname):
    # check file
    if os.path.exists(outfname):
        os.remove( outfname )
        
    chan = emirt.emio.imread(infname)
    
    chan = emirt.volume_util.norm( chan )
    
    f = h5py.File( outfname )
    f.create_dataset("/main", data=chan)
    f.close()

def segment(infname, outfname):
    # check file
    if os.path.exists( outfname ):
        os.remove( outfname )
    # read tif
    seg = emirt.emio.imread( infname )
    # transform image type    
    if seg.dtype=='uint8' and seg.shape[3]==3:
        # read the VAST output RGB images
        seg = seg.astype('uint32')
        seg = seg[:,:,:,0]*256*256 + seg[:,:,:,1]*256 + seg[:,:,:,2]
    else:
        seg = seg.astype('uint32')
                
    sids = np.unique(seg)
    print "segment ids: ", sids
    assert(len(sids)>2)
    # fake 
    dend = np.array([sids[-1],sids[-2]]).reshape(2,1).astype('uint32')
    dendValues = np.array([0.001]).astype('float32')
    #%% write
    f = h5py.File( outfname, 'w')
    f.create_dataset('/dend',data= dend)
    f.create_dataset('/dendValues', data= dendValues)
    f.create_dataset('/main', data=seg)
    f.close()

if __name__ == '__main__':
    import sys
    assert( len(sys.argv)>2 )
    print "usage: \npython fake_seg.py raw_tif_file label_tif_file"
    if len(sys.argv)==5:
        chann(   sys.argv[1], sys.argv[2] )
        segment( sys.argv[3], sys.argv[4] )
    elif len(sys.argv)==3:
        chann(   sys.argv[1], "chann.h5" )
        segment( sys.argv[2], "seg.h5" )
    import os
    os.system("bash omnifycation.sh")