# test the malis of boundary map
import emirt
import numpy as np
import time
import utils
import matplotlib.pylab as plt

def make_unique_bdm(bdm):
    bdm -= np.arange(bdm.size).reshape(bdm.shape) * 0.0001
    return bdm


def make_fake_2D_bdm(r1, r2, c1, c2):
    # fake image size
    fs = 10
    bdm = np.ones((fs,fs), dtype='float32')
    bdm[r1,:] = 0.5
    bdm[r1,7] = 0.8
    bdm[r1,r1] = 0.2
    bdm[r2,:] = 0.5
    bdm[r2,c1] = 0.2
    bdm[r2,c2] = 0.8
    lbl = np.zeros((fs,fs), dtype='uint32')
    lbl[:r2, :] = 1
    lbl[r2+1:, :] = 2
    assert lbl.max()>1
    return bdm, lbl

def make_fake_3D_aff( r1, r2, c1, c2 ):
    lbl = np.zeros((2,10,10), dtype='uint32')
    lbl[0, :r2,   :] = 1
    lbl[0, r2+1:, :] = 2
    lbl[1, r
        :r2-1, :] = 1
    lbl[1, r2:,   :] = 2

    # initialized as true affinity
    affs = emirt.volume_util.seg2aff( lbl )

    # add x splitter
    #affs[2, :, r1, : ] = 0.5
    #affs[2, :, r1, r1] = 0.2
    #affs[2, :, r1, r2] = 0.8


    # add y splitter
    #affs[1, :, r1, : ] = 0.5
    #affs[1, :, r1, r1] = 0.2
    #affs[1, :, r1, r2] = 0.8

    # add x merger
    #affs[2, :, r2-1, : ] = 0.5
    #affs[2, :, r2-1, r1] = 0.2
    #affs[2, :, r2-1, r2] = 0.8

    # add y merger
    #affs[1, :, r2-1, : ] = 0.5
    #affs[1, :, r2-1, r1] = 0.2
    #affs[1, :, r2-1, r2] = 0.8

    #affs[1, :, r2, : ] = 0.5
    #affs[1, :, r2, r1] = 0.2
    #affs[1, :, r2, r2] = 0.8

    # add z merger
    affs[0, 1, r2-2:r2+1, : ] = 0.5
    affs[0, 1, r2-2:r2+1, r1] = 0.2
    affs[0, 1, r2-2:r2+1, r2] = 0.8

    affs[0, 1, r2, : ] = 0.5
    affs[0, 1, r2, r1] = 0.2
    affs[0, 1, r2, r2] = 0.8

    return affs, lbl

def read_image(pars):
    z=8
    is_fake = pars['is_fake']
    #%% read images
    if not is_fake:
        bdm = emirt.emio.imread('../../dataset/malis/out_sample91_output_0.tif')
        lbl = emirt.emio.imread('../../dataset/malis/Merlin_label2_24bit.tif')
        raw = emirt.emio.imread('../../dataset/malis/Merlin_raw2.tif')
        lbl = emirt.volume_util.lbl_RGB2uint32(lbl)
        lbl = lbl[z,:,:]
        bdm = bdm[z,:,:]


    # only a corner for test
    corner_size = pars['corner_size']
    if corner_size > 0:
        lbl = lbl[:corner_size, :corner_size]
        bdm = bdm[:corner_size, :corner_size]

    # fill label holes
    print "fill boundary hole..."
    utils.fill_boundary_holes( lbl )

    # increase boundary width
    erosion_size = pars['erosion_size']
    if erosion_size>0:
        print "increase boundary width"
        erosion_structure = np.ones((erosion_size, erosion_size))
        msk = np.copy(lbl>0)
        from scipy.ndimage.morphology import binary_erosion
        msk = binary_erosion(msk, structure=erosion_structure)
        lbl[msk==False] = 0

    print "boundary map: ", bdm

    return bdm, lbl

if __name__ == "__main__":
    from malis_test import get_params
    pars = get_params()
    if pars['is_affinity']:
        affs, lbl = make_fake_3D_aff( 3, 7, 3, 7 )
    else:
        bdm, lbl = read_image(pars)

    # normal boundary map and
    bdm.astype("float64").tofile("../dataset/malis/bdm.bin")
    lbl.astype('float64').tofile("../dataset/malis/lbl.bin")

    from cost_fn import constrain_label
    mbdm, sbdm = constrain_label(bdm, lbl)
    mbdm.astype("float64").tofile("../dataset/malis/bdm_merge.bin")
    sbdm.astype("float64").tofile("../dataset/malis/bdm_splite.bin")
