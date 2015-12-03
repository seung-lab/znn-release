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


def exchange_x_z( affs_in ):
    affs = np.copy( affs_in )
    tmp = np.copy( affs[0,:,:,:] )
    affs[0,:,:,:] = np.copy( affs[2,:,:,:] )
    affs[2,:,:,:] = np.copy( tmp )
    return affs

def aleks_malis(affs, lbl):
    import pymalis

    true_affs = emirt.volume_util.seg2aff( lbl )
    # exchange x and z
    affs = exchange_x_z( affs )
    true_affs = exchange_x_z( true_affs )

    affs = np.ascontiguousarray( affs.astype('float32') )
    true_affs = np.ascontiguousarray( true_affs.astype('float32') )

    print "input affinity: ", affs
    print "true affinity: ", true_affs

    me, se, re, num_non_bdr = pymalis.zalis(  affs, true_affs, 1.0, 0.0, 0)

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

def make_edge_unique( affs ):
    """
    make xy affinity edge unique
    """
    step = 0.0
    # y affinity
    for z in xrange( affs.shape[1] ):
        for y in xrange( 1, affs.shape[2] ):
            for x in xrange( affs.shape[3] ):
                step += 1
                affs[1,z,y,x] -= 0.00001 * step
    # x affinity
    for z in xrange( affs.shape[1] ):
        for y in xrange( affs.shape[2] ):
            for x in xrange( 1, affs.shape[3] ):
                step += 1
                affs[2,z,y,x] -= 0.00001 * step
    return affs


if __name__ == "__main__":
    # get the parameters
    pars = get_params()

    import data_prepare
    if pars['is_affinity']:
        if pars['is_fake']:
            #        data, lbl = data_prepare.make_fake_3D_aff( 3, 7, 3, 7)
            data, lbl = data_prepare.make_fake_2D_bdm( 3,7, 3, 7 )
            # transform boundary map to affinity
            data = emirt.volume_util.bdm2aff( data )
        else:
            data, lbl = data_prepare.read_image(pars)
            data = emirt.volume_util.bdm2aff( data )

    if pars['is_affinity']:
        # transform to affinity map
        true_affs = emirt.volume_util.seg2aff( lbl.reshape((1,)+lbl.shape) )
        lbl2 = emirt.volume_util.aff2seg( true_affs )
        print "original label: ", lbl
        print "transformed lable: ", lbl2
        if pars['is_fake']:
            data = make_edge_unique( data )

    import cost_fn
    start = time.time()
    if pars['is_constrained']:
        if pars['is_aleks']:
            w, me, se = constrained_aleks_malis(data, lbl)
        else:
            print "compute the constrained malis weight..."
            w, me, se = cost_fn.constrained_malis_weight_bdm_2D(data, lbl, \
                                                                is_affinity = pars['is_affinity'])
    else:
        print "normal malis with aleks version..."
        w, me, se = aleks_bin_malis(data, lbl)

        # python interface of malis
        w2, me2, se2 = aleks_malis( data, lbl )

        print "me: ", me
        print "me2: ", me2

        print "se: ", se
        print "se2: ", se2

        assert( np.all(se2==se) )
        assert( np.all(me2==me) )

        me = exchange_x_z( me )
        se = exchange_x_z( se )

        w, me3, se3 = cost_fn.malis_weight_aff(data, lbl)
        print "python me: ", me3
        print "python se: ", se3
        assert ( np.all( me==me3 ) )
        assert ( np.all( se==se3 ) )

    elapsed = time.time() - start
    print "elapsed time is {} sec".format(elapsed)

    import malis_show
    malis_show.plot(pars, data, lbl, me, se)

    print "------end-----"
