#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import emirt
import utils
from malis.pymalis import zalis


def get_cls(props, lbls, mask=None):
    """
    compute classification error.

    Parameters
    ----------
    props : dict of array, network propagation output volumes.
    lbls  : dict of array, ground truth

    Returns
    -------
    c : number of classification error
    """
    errors = dict()
    c = 0.0

    #Applying mask if it exists
    props = utils.mask_dict_vol(props, mask)
    lbls = utils.mask_dict_vol(lbls, mask)

    for name, prop in props.iteritems():
        lbl = lbls[name]
        c += np.count_nonzero( (prop>0.5) != (lbl>0.5) )

    return c

#@jit(nopython=True)
def square_loss(props, lbls, mask=None):
    """
    compute square loss

    Parameters
    ----------
    props: numpy array, forward pass output
    lbls:  numpy array, ground truth labeling

    Return
    ------
    err:   cost energy
    grdts: numpy array, gradient volumes
    """
    grdts = dict()
    err = 0

    #Applying mask if it exists
    props = utils.mask_dict_vol(props, mask)
    lbls = utils.mask_dict_vol(lbls, mask)

    for name, prop in props.iteritems():
        lbl = lbls[name]

        grdt = prop - lbl
        grdts[name] = grdt * 2

        err += np.sum(np.square( grdt ))

    return (props, err, grdts)

def square_square_loss(props, lbls, mask=None, margin=0.2):
    """
    square-square loss (square loss with a margin)
    """
    gradients = dict()
    error = 0

    #Applying mask if it exists
    props = utils.mask_dict_vol(props, mask)
    lbls = utils.mask_dict_vol(lbls, mask)

    for name, propagation in props.iteritems():
        lbl = lbls[name]

        gradient = propagation - lbl
        gradient[np.abs(gradient) <= margin] = 0
        gradients[name] = gradient * 2

        error += np.sum(np.square( gradient ))


    return (props, error, gradients)

#@jit(nopython=True)
def binomial_cross_entropy(props, lbls, mask=None):
    """
    compute binomial cost

    Parameters
    ----------
    props:  dict of network output arrays
    lbls:   dict of ground truth arrays

    Return
    ------
    err:    cost energy
    grdts:  dict of gradient volumes
    """
    grdts = dict()
    err = 0

    #Taking a slightly different strategy with masking
    # to improve the numerical stability of the error output
    entropy = dict()

    #Finding Gradients
    for name, prop in props.iteritems():
        lbl = lbls[name]

        grdts[name] = prop - lbl

        entropy[name] = -lbl*np.log(prop) - (1-lbl)*np.log(1-prop)

    #Applying mask if it exists
    grdts = utils.mask_dict_vol(grdts, mask)
    entropy = utils.mask_dict_vol(entropy, mask)

    for name, vol in entropy.iteritems():
        err += np.sum( vol )

    return (props, err, grdts)


#@jit(nopython=True)
def softmax(props):
    """
    softmax activation

    Parameters:
    props:  numpy array, net forward output volumes

    Returns:
    ret:   numpy array, softmax activation volumes
    """
    ret = dict()
    for name, prop in props.iteritems():
        # make sure that it is the output of binary class
        # assert(prop.shape[0]==2)

        # rebase the prop for numerical stability
        # mathematically, this does not affect the softmax result!
        propmax = np.max(prop, axis=0)
        for c in xrange( prop.shape[0] ):
            prop[c,:,:,:] -= propmax

        prop = np.exp(prop)
        pesum = np.sum(prop, axis=0)
        ret[name] = np.empty(prop.shape, dtype=prop.dtype)
        for c in xrange(prop.shape[0]):
            ret[name][c,:,:,:] = prop[c,:,:,:] / pesum
    return ret

def multinomial_cross_entropy(props, lbls, mask=None):
    """
    compute multinomial cross entropy

    Parameters
    ----------
    props:    list of forward pass output
    lbls:     list of ground truth labeling

    Return
    ------
    err:    cost energy
    cls:    classfication error
    grdts:  list of gradient volumes
    """
    grdts = dict()
    err = 0

    #Taking a slightly different strategy with masking
    # to improve the numerical stability of the error output
    entropy = dict()

    for name, prop in props.iteritems():
        lbl = lbls[name]

        grdts[name] = prop - lbl

        entropy[name] = -lbl * np.log(prop)

    #Applying mask if it exists
    grdts = utils.mask_dict_vol(grdts, mask)
    entropy = utils.mask_dict_vol(entropy, mask)

    for name, vol in entropy.iteritems():
        err += np.sum( vol )

    return (props, err, grdts)

def softmax_loss(props, lbls, mask=None):
    props = softmax(props)
    return multinomial_cross_entropy(props, lbls, mask)

def softmax_loss2(props, lbls, mask=None):
    grdts = dict()
    err = 0

    for name, prop in props.iteritems():
        # make sure that it is the output of binary class
        assert(prop.shape[0]==2)

        print "original prop: ", prop

        # rebase the prop for numerical stability
        # mathimatically, this do not affect the softmax result!
        # http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
#        prop = prop - np.max(prop)
        propmax = np.max(prop, axis=0)
        prop[0,:,:,:] -= propmax
        prop[1,:,:,:] -= propmax

        log_softmax = np.empty(prop.shape, dtype=prop.dtype)
        log_softmax[0,:,:,:] = prop[0,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
        log_softmax[1,:,:,:] = prop[1,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
        prop = np.exp(log_softmax)
        props[name] = prop

        lbl = lbls[name]
        grdts[name] = prop - lbl
        err = err + np.sum( -lbl * log_softmax )
        print "gradient: ", grdts[name]
        assert(not np.any(np.isnan(grdts[name])))
    return (props, err, grdts)

#def hinge_loss(props, lbls):
# TO-DO


def malis_weight_aff(affs, true_affs):
    """
    compute malis tree_size

    Parameters:
    -----------
    affs:      4D array of forward pass output affinity graphs, size: C*Z*Y*X
    true_affs : 4d array of ground truth affinity graph
    Dim: dimension
    threshold: threshold for segmentation

    Return:
    ------
    weights : 4D array of weights
    """
    # segment the true affinity graph
    if true_affs.shape[0]==3:
        tseg = emirt.volume_util.aff2seg(true_affs)
    elif true_affs.ndim==3 or (true_affs.ndim==4 and true_affs.shape[0]==1):
        tseg = true_affs
    elif true_affs.ndim==2:
        tseg = np.reshape( true_affs, (1,)+true_affs.shape )
    else:
        print "ground truth shape: ", true_affs.shape
        raise NameError( 'invalid true_affs shape' )

    print "true segmentation: ", tseg

    # get affinity graphs
    xaff = affs[2,:,:,:]
    yaff = affs[1,:,:,:]
    zaff = affs[0,:,:,:]

    # voxel ids
    vids = np.arange( xaff.size, dtype='uint32' ).reshape( xaff.shape )

    # initialize edges: aff, id1, id2, z/y/x, true_aff
    edges = list()
    # x affinity edge
    for z in xrange( xaff.shape[0] ):
        for y in xrange( xaff.shape[1] ):
            for x in xrange( 1, xaff.shape[2] ):
                edges.append( (xaff[z,y,x], vids[z,y,x], vids[z,y,x-1], 2,z,y,x) )
    # y affinity edge
    for z in xrange( yaff.shape[0] ):
        for y in xrange( 1, yaff.shape[1] ):
            for x in xrange( yaff.shape[2] ):
                edges.append( (yaff[z,y,x], vids[z,y,x], vids[z,y-1,x], 1,z,y,x) )

    # z affinity edge
    for z in xrange( 1, zaff.shape[0] ):
        for y in xrange( zaff.shape[1] ):
            for x in xrange( zaff.shape[2] ):
                edges.append( (zaff[z,y,x], vids[z,y,x], vids[z-1,y,x], 0,z,y,x) )
    # descending sort
    edges.sort(reverse=True)

    # find the maximum-spanning tree based on union-find algorithm
    merr = np.zeros( affs.shape, dtype=affs.dtype )
    serr = np.zeros( affs.shape, dtype=affs.dtype )

    # initialize the watershed domains
    dms = emirt.domains.CDomains( tseg )

    # union find the sets
    for e in edges:
        # voxel ids
        vid1 = e[1]
        vid2 = e[2]
        c = e[-4]
        z = e[-3]
        y = e[-2]
        x = e[-1]
        # union the domains
        me, se = dms.union( vid1, vid2 )

        # deal with the maiximin edge
        # accumulate the merging error
        merr[c,z,y,x] += me
        serr[c,z,y,x] += se

    # combine the two error weights
    w = (merr + serr)
    return w, merr, serr


def malis_weight_bdm_2D(bdm, lbl, threshold=0.5):
    """
    compute malis weight for boundary map

    Parameters
    ----------
    bdm: 2D array, forward pass output boundary map
    lbl: 2D array, manual labels containing segment ids
    threshold: binarization threshold

    Returns
    -------
    weights: 2D array of weights
    """
    # eliminate the second output
    assert(bdm.ndim==2)
    assert(bdm.shape==lbl.shape)

    # initialize segmentation with individual id of each voxel
    # voxel id start from 0, is exactly the coordinate of voxel in 1D
    vids = np.arange(bdm.size).reshape( bdm.shape )

    # create edges: bdm, id1, id2, true label
    # the affinity of neiboring boundary map voxels
    # was represented by the minimal boundary map value

    edges = list()
    for y in xrange(bdm.shape[0]):
        for x in xrange(bdm.shape[1]-1):
            bmv1 = bdm[y,x]
            vid1 = vids[y,x]
            bmv2 = bdm[y,x+1]
            vid2 = vids[y,x+1]
            # the voxel with id1 will always has the minimal value
            if bmv1 > bmv2:
                bmv1, bmv2 = bmv2, bmv1
                vid1, vid2 = vid2, vid1
            edges.append((bmv1, vid1, vid2))

    for y in xrange(bdm.shape[1]-1):
        for x in xrange(bdm.shape[0]):
            # boundary map value and voxel id
            bmv1 = bdm[y,x]
            vid1 = vids[y,x]
            bmv2 = bdm[y+1,x]
            vid2 = vids[y+1,x]
            if bmv1 > bmv2:
                bmv1, bmv2 = bmv2, bmv1
                vid1, vid2 = vid2, vid1
            edges.append((bmv1, vid1, vid2))

    # descending sort
    edges.sort(reverse=True)

    # initalize the merge and split errors
    merr = np.zeros(bdm.size, dtype=bdm.dtype)
    serr = np.zeros(bdm.size, dtype=bdm.dtype)

    # initalize the watershed domains
    dms = emirt.domains.CDomains( lbl )

    # find the maximum spanning tree based on union-find algorithm
    for e in edges:
        # voxel ids
        vid1 = e[1]
        vid2 = e[2]
        # union the domains
        me, se = dms.union( vid1, vid2 )

        # deal with the maximin edge
        # accumulate the merging error
        merr[vid1] += me
        # accumulate the spliting error
        serr[vid1] += se

    # reshape the err
    merr = merr.reshape(bdm.shape)
    serr = serr.reshape(bdm.shape)
    # combine the two error weights
    w = (merr + serr)

    return (w, merr, serr)

def constrain_label(prp, lbl):
    """
    parameters:
    -----------
    prp: 4D array, forward propagation result, could be boundary or aff map
    lbl: 4D array, ground truth labeling
    """
    assert prp.shape == lbl.shape
    # merging error boundary map filled with intracellular ground truth
    mprp = np.copy(prp)
    mprp[lbl>0] = 1

    # splitting error boundary map filled with boundary ground truth
    sprp = np.copy(prp)
    sprp[lbl==0] = 0
    return mprp, sprp

def constrained_malis(prp, lbl, threshold=0.5):
    """
    adding constraints for malis weight
    fill the intracellular space with ground truth when computing merging error
    fill the boundary with ground truth when computing spliting error
    """
    mprp, sprp = constrain_label(prp, lbl)
    # get the merger weights
    mme, mse, mre, num, mtp, mtn, mfp, mfn = zalis(mprp, lbl, 1.0, 0.5, 0)
    # normalization
    mme = mme / (mfp + mtn)



    # get the splitter weights
    sme, sse, sre, num, stp, stn, sfp, sfn = zalis(sprp, lbl, 0.5, 0.0, 0)
    # normalization
    sse = sse / (stp + sfn)

    re = (mfp + sfn)/(mtp+mtn+mfp+mfn)
    w = mme + sse

    #print "stp: ",stp, "  sfn: ",sfn, "  stn: ",stn,"  sfp: ",sfp
    #print "mfp: ",mfp, "  mtn: ",mtn, "  mtp: ",mtp,"  mfn: ",mfn
    #print "mprp: ",mprp
    #print "sprp: ",sprp
    #print "prp: ",prp

    return (w, mme, sse, re, num)

def constrained_malis_weight_bdm_2D(bdm, lbl, threshold=0.5):
    """
    adding constraints for malis weight
    fill the intracellular space with ground truth when computing merging error
    fill the boundary with ground truth when computing spliting error
    """
    mbdm, sbdm = constrain_label(bdm, lbl)
    # get the merger weights
    mw, mme, mse = malis_weight_bdm_2D(mbdm, lbl, threshold)
    # get the splitter weights
    sw, sme, sse = malis_weight_bdm_2D(sbdm, lbl, threshold)
    w = mme + sse
    return (w, mme, sse)

def malis_weight_bdm(bdm, lbl, threshold=0.5):
    """
    compute the malis weight of boundary map

    Parameter
    ---------
    bdm: 3D or 4D array, boundary map
    lbl: 3D or 4D array, binary ground truth

    Return
    ------
    weights: 3D or 4D array, the malis weights
    merr: merger error
    serr: splitter error
    """
    assert(bdm.shape==lbl.shape)
    assert(bdm.ndim==4 or bdm.ndim==3)
    original_shape = bdm.shape
    if bdm.ndim==3:
        bdm = bdm.reshape((1,)+(bdm.shape))
        lbl = lbl.reshape((1,)+(lbl.shape))

    # only compute weight of the first channel
    bdm0 = bdm[0,:,:,:]
    # segment the ground truth label
    lbl0 = emirt.volume_util.bdm2seg(lbl[0,:,:,:])

    # initialize the weights
    weights = np.empty(bdm.shape, dtype=bdm.dtype)
    merr = np.empty(bdm.shape, dtype=bdm.dtype)
    serr = np.empty(bdm.shape, dtype=bdm.dtype)

    # traverse along the z axis
    for z in xrange(bdm.shape[1]):
        w, me, se = malis_weight_bdm_2D(bdm0[z,:,:], lbl0[z,:,:], threshold)
        for c in xrange(bdm.shape[0]):
            weights[c,z,:,:] = w
            merr[c,z,:,:] = me
            serr[c,z,:,:] = se
    weights = weights.reshape( original_shape )
    merr = merr.reshape( original_shape )
    serr = serr.reshape( original_shape )
    return weights, merr, serr

def malis_weight(pars, props, lbls):
    """
    compute the malis weight including boundary map and affinity cases
    """
    malis_weights = dict()
    rand_errors = dict()

    # malis normalization type
    if 'frac' in pars['malis_norm_type']:
        is_frac_norm = 1
    else:
        is_frac_norm = 0

    for name, prop in props.iteritems():
        assert prop.ndim==4
        lbl = lbls[name]
        if prop.shape[0]==3:
            if 'constrain' in pars['malis_norm_type']:
                mw, merr, serr, re, num_non_bdr = constrained_malis(prop, lbl)
            else:
                # affinity output
                merr, serr, re, num_non_bdr, \
                    tp, tn, fp, fn = zalis( prop, lbl, \
                                            1.0, 0.0, is_frac_norm)
                print "tp: ",tp," tn: ",tn, "  fp:",fp,"  fn:",fn
            mw = merr + serr
            # normalization
            if 'num' in pars['malis_norm_type']:
                mw = mw / float(num_non_bdr)
            elif 'pair' in pars['malis_norm_type']:
                mw = mw / float(num_non_bdr * (num_non_bdr-1))
            if pars['is_malis_renorm']:
                # normalize the malis weight to make the sum to be N'
                mw = mw * ( mw.size / np.sum(mw) )

            malis_weights[name] = mw
            rand_errors[name] = re
        else:
            # take it as boundary map
            malis_weights[name], merr, serr = malis_weight_bdm(prop, lbl)

    return (malis_weights, rand_errors, num_non_bdr)
