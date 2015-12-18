# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 14:27:31 2015

@author: jingpeng
"""

import numpy as np

#%% read hdf5 volume
def imread( fname ):
    if '.hdf5' in fname or '.h5' in fname:
        import h5py
        f = h5py.File( fname )
        v = np.asarray( f['/main'] )
        f.close()
        print 'finished reading image stack :)'
        return v
    elif '.tif' in fname:
#        import skimage.io
#        vol = skimage.io.imread( fname, plugin='tifffile' )
        import tifffile
        vol = tifffile.imread(fname)
        return vol
    else:
        print "read as znn image..."
        return znn_img_read(fname)


def imsave( vol, fname ):
    if '.hdf5' in fname or '.h5' in fname:
        import h5py
        f = h5py.File( fname )
        f.create_dataset('/main', data=vol, compression="gzip")
        f.close()
    elif '.tif' in fname:
#        import skimage.io
#        skimage.io.imsave(fname, vol, plugin='tifffile')
        import tifffile
        tifffile.imsave(fname, vol)
    else:
        print "save as znn image..."
        znn_img_save(vol, fname)

# load binary znn image
def znn_img_read( fname ):
    if '.image' in fname:
        fname = fname.replace('.image', "")
        ext = ".image"
    elif '.label' in fname:
        fname = fname.replace(".label", "")
        ext = ".label"
    else:
        ext = ""
    sz = np.fromfile(fname+'.size', dtype='uint32')[::-1]
    vol = np.fromfile(fname + ext, dtype='double').reshape(sz)
    return vol

def znn_img_save(vol, fname, dtype = 'double'):
    vol = vol.astype('double')
    if ".image" in fname:
        fname = fname.replace(".image", "")
        ext = ".image"
    elif ".label" in fname:
        fname = fname.replace(".label", "")
        ext = ".label"
    else:
        ext = ""
    vol.tofile(fname+ext)
    sz = np.asarray( vol.shape, dtype='uint32' )[::-1]
    sz.tofile(fname+".size")

def tif2h5(intif, outh5):
    from tifffile import TiffFile
    f = h5py.File( outh5 )
    # how to get tif shape?
    f.create_dataset('/main', shape=())
    hv = f['/main']
    with TiffFile(infname) as tif:
        for k,page in enumerate(tif):
            hv[k] = page.asarray()
    f.close()

def write_for_znn(Dir, vol, cid):
    '''transform volume to znn format'''
    # make directory
    import emirt.os
    emirt.os.mkdir_p(Dir )
    emirt.os.mkdir_p(Dir + 'data')
    emirt.os.mkdir_p(Dir + 'spec')
    vol.tofile(Dir + 'data/' + 'batch'+str(cid)+'.image')
    sz = np.asarray(vol.shape)
    sz.tofile(Dir + 'data/' + 'batch'+str(cid)+'.size')

    # printf the batch.spec
    f = open(Dir + 'spec/' + 'batch'+str(cid)+'.spec', 'w')
    f.write('[INPUT1]\n')
    f.write('path=./dataset/piriform/data/batch'+str(cid)+'\n')
    f.write('ext=image\n')
    f.write('size='+str(sz[2])+','+str(sz[1])+','+str(sz[0])+'\n')
    f.write('pptype=standard2D\n\n')

def h5write( fname, data_path, data, compression="gzip" ):
    """
    save dataset in hdf5 file
    """
    import h5py
    f = h5py.File( fname, 'a' )
    f.create_dataset(data_path, data=data, compression=compression)
    f.close()

def h5read( fname, data_path ):
    """
    read dataset in hdf5 file
    """
    import h5py
    f = h5py.File( fname )
    data = f[data_path].value
    f.close()
    return data
