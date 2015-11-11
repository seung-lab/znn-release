# test the malis of boundary map
import sys
sys.path.append('../')

import numpy as np
import time
import emirt
#%% parameters
def get_params():
    pars = dict()
    # epsilone: a small number for log to avoind -infinity
    pars['eps'] = 0.0000001

    # largest disk radius
    pars['Dm'] = 500
    pars['Ds'] = 500

    # use affinity map or not
    pars['is_affinity'] = True

    # make a fake test image
    pars['is_fake'] = True

    # use aleks malis
    pars['is_aleks'] = True

    # whether using constrained malis
    pars['is_constrained'] = False

    # thicken boundary of label by morphological errosion
    pars['erosion_size'] = 0

    # a small corner
    pars['corner_size'] = 0

    # disk radius threshold
    pars['DrTh'] = 0
    return pars


def exchange_x_z( affs ):
    tmp = np.copy( affs[0,:,:,:] )
    affs[0,:,:,:] = np.copy( affs[2,:,:,:] )
    affs[2,:,:,:] = np.copy( tmp )
    return affs

def aleks_malis(affs, true_affs):
    import pymalis

    # exchange x and z
    affs = exchange_x_z( affs )
    true_affs = exchange_x_z( true_affs )

    affs = np.ascontiguousarray( affs.astype('float32') )
    true_affs = np.ascontiguousarray( true_affs.astype('float32') )

    print "input affinity: ", affs
    print "true affinity: ", true_affs

    me, se = pymalis.zalis(  affs, true_affs, 1.0, 0.0, 1 )

    # total error
    w = me + se
    return w, me, se

def aleks_bin_malis( affs, lbl ):
    affs = np.ascontiguousarray( affs.astype('float64') )
    lbl = np.ascontiguousarray( lbl.astype('float64') )
    # save as binary file
    affs[0,:,:,:].tofile('../../experiments/malis/zaff.bin')
    affs[1,:,:,:].tofile('../../experiments/malis/yaff.bin')
    affs[2,:,:,:].tofile('../../experiments/malis/xaff.bin')

#    lbl = emirt.volume_util.aff2seg( true_affs )
    lbl.tofile( '../../experiments/malis/label.bin' )

    # run binary
    import os
    os.system( 'cd ../../; ./bin/malis malis.options' )

    # read the output
    me = np.fromfile( '../../experiments/malis/out.merger', dtype='float64' ).reshape(affs.shape)
    se = np.fromfile( '../../experiments/malis/out.splitter', dtype='float64' ).reshape(affs.shape)
    w = me + se
    return w, me, se

def constrained_aleks_malis(affs, lbl, threshold=0.5):
    """
    adding constraints for malis weight
    fill the intracellular space with ground truth when computing merging error
    fill the boundary with ground truth when computing spliting error
    """
    from cost_fn import constrain_label
    mbdm, sbdm = constrain_label(bdm, lbl)
    # get the merger weights
    mw, mme, mse = aleks_malis(mbdm, lbl)
    # get the splitter weights
    sw, sme, sse = aleks_malis(sbdm, lbl)
    w = mme + sse
    return (w, mme, sse)

if __name__ == "__main__":
    # get the parameters
    pars = get_params()

    import data_prepare
    data, lbl = data_prepare.read_image(pars)

    if pars['is_affinity']:
        # transform to affinity map
        data = emirt.volume_util.bdm2aff( data )
        true_affs = emirt.volume_util.seg2aff( lbl.reshape((1,)+lbl.shape) )

    # recompile and use cost_fn
    #print "compile the cost function..."
    #os.system('python compile.py cost_fn')
    import cost_fn
    start = time.time()
    if pars['is_constrained']:
        if pars['is_aleks']:
            w, me, se = constrained_aleks_malis(data, lbl)
        else:
            print "compute the constrained malis weight..."
            w, me, se = cost_fn.constrained_malis_weight_bdm_2D(data, lbl, is_affinity = pars['is_affinity'])
    else:
        if pars['is_aleks']:
            print "normal malis with aleks version..."
            w, me, se = aleks_bin_malis(data, lbl)
        else:
            print "normal malis weight with python version..."
            print "true_affs: ", true_affs
            print "affs: ", data
            w, me, se = cost_fn.malis_weight_aff(data, true_affs, Dim=2)

    elapsed = time.time() - start
    print "elapsed time is {} sec".format(elapsed)

    print "merger error: ", me
    print "splitter error: ", se

    import malis_show
    malis_show.plot(pars, data, lbl, me, se)

    print "------end-----"
