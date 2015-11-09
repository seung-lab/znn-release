# test the malis of boundary map
import emirt
import numpy as np
import time
import utils
import matplotlib.pylab as plt

def make_unique_bdm(bdm):
    bdm -= np.arange(bdm.size).reshape(bdm.shape) * 0.0001
    return bdm

def read_image(pars):
    z=8
    is_fake = pars['is_fake']
    #%% read images
    if not is_fake:
        bdm = emirt.emio.imread('../dataset/out_sample91_output_0.tif')
        lbl = emirt.emio.imread('../dataset/Merlin_label2_24bit.tif')
        raw = emirt.emio.imread('../dataset/Merlin_raw2.tif')
        lbl = emirt.volume_util.lbl_RGB2uint32(lbl)
        lbl = lbl[z,:,:]
        bdm = bdm[z,:,:]
    else:
        # fake image size
        fs = 10
        bdm = np.ones((fs,fs), dtype='float32')
        bdm[3,:] = 0.5
        bdm[3,7] = 0.8
        bdm[3,3] = 0.2
        bdm[7,:] = 0.5
        bdm[7,3] = 0.2
        bdm[7,7] = 0.8
        lbl = np.zeros((fs,fs), dtype='uint32')
        lbl[:7, :] = 1
        lbl[8:, :] = 2
        assert lbl.max()>1

    # make unique
    # bdm = make_unique_bdm( bdm )

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
    bdm, lbl = read_image(pars)

    # normal boundary map and
    bdm.astype("float64").tofile("../dataset/boundary_map.bin")
    lbl.astype('float64').tofile("../dataset/label.bin")

    from cost_fn import constrain_label
    mbdm, sbdm = constrain_label(bdm, lbl)
    mbdm.astype("float64").tofile("../dataset/bdm_merge.bin")
    sbdm.astype("float64").tofile("../dataset/bdm_splite.bin")
