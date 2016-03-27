#!/usr/bin/env python
__doc__ = """

Dataset Class Interface (CSamples)

Jingpeng Wu <jingpeng.wu@gmail.com>,
Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""

import sys
import numpy as np
import emirt
import utils
from zdataset import *
import os
import h5py

class CSample(object):
    """
    Sample Class, which represents a pair of input and output volume structures
    (as CInputImage and COutputImage respectively)

    Allows simple interface for procuring matched random samples from all volume
    structures at once

    """
    def __init__(self, config, pars, sample_id, net, \
                 outsz, setsz_ins=None, setsz_outs=None,\
                 log=None, is_forward=False):

        # Parameter object (dict)
        self.pars = pars

        self.sid = sample_id
        # Name of the sample within the configuration file
        # Also used for logging
        self.sec_name = "sample%d" % sample_id

        # temporary layer names
        if is_forward and setsz_ins is None and setsz_outs is None:
            print "forward pass, get setsz from network"
            self.setsz_ins  = net.get_inputs_setsz()
            self.setsz_outs = net.get_outputs_setsz()
        else:
            self.setsz_ins  = setsz_ins
            self.setsz_outs = setsz_outs
        fov = np.asarray(net.get_fov(), dtype='uint32')

        # Loading input images
        print "\ncreate input image class..."
        self.imgs = dict()
        self.ins = dict()
        for name,setsz_in in self.setsz_ins.iteritems():
            #Finding the section of the config file
            imid = config.getint(self.sec_name, name)
            imsec_name = "image%d" % (imid,)
            self.ins[name] = ConfigInputImage( config, pars, imsec_name, \
                                      outsz, setsz_in, fov, is_forward=is_forward )
            self.imgs[name] = self.ins[name].data

        self.lbls = dict()
        self.msks = dict()
        self.outs = dict()

        if not is_forward:
            print "\ncreate label image class..."
            for name,setsz_out in self.setsz_outs.iteritems():
                #Allowing for users to abstain from specifying labels
                if not config.has_option(self.sec_name, name):
                    continue
                #Finding the section of the config file
                imid = config.getint(self.sec_name, name)
                imsec_name = "label%d" % (imid,)
                self.outs[name] = ConfigOutputLabel( config, pars, imsec_name, \
                                                     outsz, setsz_out, fov)
                self.lbls[name] = self.outs[name].data
                self.msks[name] = self.outs[name].msk

        if not is_forward:
            self._prepare_training()

        #Filename for log
        self.log = log

    def get_dataset(self):
        raw = self.ins.values()[0].get_dataset()
        lbl = self.outs.values()[0].get_dataset()
        return raw, lbl

    def _prepare_training(self):
        """
        prepare data for training
        """
        # init deviation range
        # we need to consolidate this over all input and output volumes
        dev_high = np.array([sys.maxsize, sys.maxsize, sys.maxsize])
        dev_low  = np.array([-sys.maxint-1, -sys.maxint-1, -sys.maxint-1])

        for name,setsz in self.setsz_ins.iteritems():
            low, high = self.ins[name].get_dev_range()
            # Deviation bookkeeping
            dev_high = np.minimum( dev_high, high )
            dev_low  = np.maximum( dev_low , low  )

        # define output images
        for name, setsz in self.setsz_outs.iteritems():
            low, high = self.outs[name].get_dev_range()
            # Deviation bookkeeping
            dev_high = np.minimum( dev_high, high )
            dev_low  = np.maximum( dev_low , low  )

        # find the candidate central locations of sample
        if len(self.outs) > 0:
            # this will not work with multiple output layers!!
            self.locs = self.outs.values()[0].get_candidate_loc( dev_low, dev_high )
        else:
            print "\nWARNING: No output volumes defined!\n"
            self.locs = None

    def _data_aug(self, subinputs, subtlbls, submsks):
        # random transformation roll
        if self.pars['is_data_aug']:
            rft = (np.random.rand(4)>0.5)
            for key, subinput in subinputs.iteritems():
                subinputs[key] = utils.data_aug_transform(subinput,      rft )
            for key, subtlbl in subtlbls.iteritems():
                subtlbls[key]  = utils.data_aug_transform(subtlbl, rft )
                submsks[key]   = utils.data_aug_transform(submsks[key],  rft )
        return subinputs, subtlbls, submsks

    def get_random_sample(self):
        '''Fetches a matching random sample from all input and output volumes'''

        # random deviation
        ind = np.random.randint( np.size(self.locs[0]) )
        loc = np.empty( 3, dtype=np.uint32 )
        loc[0] = self.locs[0][ind]
        loc[1] = self.locs[1][ind]
        loc[2] = self.locs[2][ind]
        dev = loc - self.outs.values()[0].center

        self.write_request_to_log(loc)

        # get input and output 4D sub arrays
        subinputs = dict()
        for key, img in self.ins.iteritems():
            subinputs[key] = self.ins[key].get_subvolume(dev)

        subtlbls = dict()
        submsks  = dict()
        for key, lbl in self.outs.iteritems():
            subtlbls[key], submsks[key] = self.outs[key].get_subvolume(dev)

        # data augmentation
        subinputs, subtlbls, submsks = self._data_aug( subinputs, subtlbls, submsks )

        return ( subinputs, subtlbls, submsks )

    def _get_balance_weight(self, arr, msk=None):
        mask_empty = msk is None or msk.size == 0
        if mask_empty:
            values = arr
        else:
            values = arr[ np.nonzero(msk) ]

        # number of nonzero elements
        pn = float( np.count_nonzero(values) )
        # total number of elements
        num = float( np.size(values) )
        # number of zero elements
        zn = num - pn

        if pn==0 or zn==0:
            return 1,1
        else:
            # weight of positive and zero
            wp = 0.5 * num / pn
            wz = 0.5 * num / zn

            return wp, wz

    # ZNNv1 uses different normalization
    # This method is only temporary (for reproducing paper results)
    def _get_balance_weight_v1(self, arr, msk=None):
        mask_empty = msk is None or msk.size == 0
        if mask_empty:
            values = arr
        else:
            values = arr[ np.nonzero(msk) ]

        # number of nonzero elements
        pn = float( np.count_nonzero(values) )
        # total number of elements
        num = float( np.size(values) )
        zn = num - pn

        # weight of positive and zero
        if pn==0 or zn==0:
            return 1,1
        else:
            wp = 1 / pn
            wz = 1 / zn
            ws = wp + wz
            wp = wp / ws
            wz = wz / ws
            return wp, wz

    def get_next_patch(self):

        inputs, outputs = {}, {}

        for name, img in self.ins.iteritems():
            inputs[name] = img.get_next_patch()
        for name, img in self.outs.iteritems():
            outputs[name] = img.get_next_patch()

        return ( inputs, outputs )

    def output_volume_shape(self):

        shapes = {}

        for name, img in self.ins.iteritems():
            shapes[name] = img.output_volume_shape()

        return shapes

    def num_patches(self):

        patch_counts = {}

        for name, img in self.ins.iteritems():
            patch_counts[name] = img.num_patches()

        return patch_counts

    def write_request_to_log(self, dev):
        '''Records the subvolume requested of this sample in a log'''
        if self.log is not None:
            log_line1 = self.sec_name
            log_line2 = "subvolume: [{},{},{}] requested".format(dev[0],dev[1],dev[2])
            utils.write_to_log(self.log, log_line1)
            utils.write_to_log(self.log, log_line2)

class CAffinitySample(CSample):
    """
    sample for affinity training
    """

    def __init__(self, config, pars, sample_id, net, outsz, log=None, is_forward=False):
        self.setsz_ins  = net.get_inputs_setsz()
        self.setsz_outs = net.get_outputs_setsz()

        if not is_forward:
            # increase the shape by 1 for affinity sample
            # this will be shrinked later
            print "increase set size..."
            for key, setsz in self.setsz_ins.iteritems():
                self.setsz_ins[key][-3:] += 1
            for key, setsz in self.setsz_outs.iteritems():
                self.setsz_outs[key][-3:] += 1

        # initialize the general sample
        CSample.__init__(self, config, pars, sample_id, net, \
                         outsz, setsz_ins = self.setsz_ins, setsz_outs = self.setsz_outs, \
                         log=log, is_forward=is_forward)

        # precompute the global rebalance weights
        self.taffs = dict()
        self.tmsks = dict()
        for k, lbl in self.lbls.iteritems():
            self.taffs[k] = self._seg2aff( lbl )
            self.tmsks[k] = self._msk2affmsk( self.msks[k] )

        self._prepare_rebalance_weights( self.taffs, self.tmsks )
        return

    def _seg2aff( self, lbl ):
        """
        transform labels to affinity.
        Note that this transformation will shrink the volume size
        it is different with normal transformation keeping the size of lable
        which is defined in emirt.volume_util.seg2aff

        Parameters
        ----------
        lbl : 4D float array, label volume.
        Returns
        -------
        aff : 4D float array, affinity graph.
        """
        if np.size(lbl)==0:
            return np.array([])

        # the 3D volume number should be one
        assert( lbl.shape[0] == 1  and lbl.ndim==4 )

        aff_size = np.asarray(lbl.shape)-1
        aff_size[0] = 3

        aff = np.zeros( tuple(aff_size) , dtype=self.pars['dtype'] )

        #z-affinity
        aff[2,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0, :-1, 1:  ,1: ]) & (lbl[0,1:,1:,1:]>0)
        #y-affinity
        aff[1,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0, 1: , :-1 ,1: ]) & (lbl[0,1:,1:,1:]>0)
        #x-affinity
        aff[0,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0, 1: , 1:  ,:-1]) & (lbl[0,1:,1:,1:]>0)

        return aff

    def _msk2affmsk( self, msk ):
        """
        transform binary mask to affinity mask

        Parameters
        ----------
        msk : 4D array, one channel, binary mask for boundary map

        Returns
        -------
        ret : 4D array, 3 channel for z,y,x direction
        """
        if np.size(msk)==0:
            return msk
        C,Z,Y,X = msk.shape
        ret = np.zeros((3, Z-1, Y-1, X-1), dtype=self.pars['dtype'])

        #Z mask
        ret[2,:,:,:] = (msk[0,1:,1:,1:]>0) | (msk[0,:-1,1:,1:]>0)
        #Y mask
        ret[1,:,:,:] = (msk[0,1:,1:,1:]>0) | (msk[0,1:,:-1,1:]>0)
        #X mask
        ret[0,:,:,:] = (msk[0,1:,1:,1:]>0) | (msk[0,1:,1:,:-1]>0)

        return ret


    def _prepare_rebalance_weights(self, taffs, tmsks):
        """
        get rebalance tree_size of gradient.
        make the nonboundary and boundary region have same contribution of training.
        taffs: dict, key is layer name, value is true affinity output
        """
        self.zwps = dict()
        self.zwzs = dict()
        self.ywps = dict()
        self.ywzs = dict()
        self.xwps = dict()
        self.xwzs = dict()

        if self.pars['rebalance_mode']:
            for k, aff in taffs.iteritems():

                msk = tmsks[k] if tmsks[k].size != 0 else np.zeros((3,0,0,0))

                self.zwps[k], self.zwzs[k] = self._get_balance_weight_v1(aff[2,:,:,:], msk[2,:,:,:])
                self.ywps[k], self.ywzs[k] = self._get_balance_weight_v1(aff[1,:,:,:], msk[1,:,:,:])
                self.xwps[k], self.xwzs[k] = self._get_balance_weight_v1(aff[0,:,:,:], msk[0,:,:,:])

        return

    def _rebalance_aff(self, subtaffs, submsks):
        """
        rebalance the affinity labeling with size of (3,Z,Y,X)
        """
        if self.pars['rebalance_mode'] is None:
            return dict()
        elif 'patch' in self.pars['rebalance_mode']:
            # recompute the weights
            self._prepare_rebalance_weights( subtaffs, submsks )

        subwmsks = dict()
        for k, subtaff in subtaffs.iteritems():
            assert subtaff.ndim==4 and subtaff.shape[0]==3
            w = np.zeros(subtaff.shape, dtype=self.pars['dtype'])

            w[2,:,:,:][subtaff[2,:,:,:] >0] = self.zwps[k]
            w[1,:,:,:][subtaff[1,:,:,:] >0] = self.ywps[k]
            w[0,:,:,:][subtaff[0,:,:,:] >0] = self.xwps[k]

            w[2,:,:,:][subtaff[2,:,:,:]==0] = self.zwzs[k]
            w[1,:,:,:][subtaff[1,:,:,:]==0] = self.ywzs[k]
            w[0,:,:,:][subtaff[0,:,:,:]==0] = self.xwzs[k]
            subwmsks[k] = w

        return subwmsks

    def get_random_sample(self):
        subimgs, sublbls, submsks = super(CAffinitySample, self).get_random_sample()

        # shrink the inputs
        for key, subimg in subimgs.iteritems():
            subimgs[key] = subimg[:,1:,1:,1:]

        # transform the label to affinity
        # this operation will shrink the volume size
        subtaffs = dict()
        for key, sublbl in sublbls.iteritems():
            assert sublbl.shape[0]==1 and sublbl.ndim==4
            subtaffs[key] = self._seg2aff( sublbl )
            # make affinity mask
            submsks[key]  = self._msk2affmsk( submsks[key] )

        # affinity map rebalance
        subwmsks = self._rebalance_aff( subtaffs, submsks )

        return subimgs, subtaffs, submsks, subwmsks

class CBoundarySample(CSample):
    """
    sample for boundary map training
    """
    def __init__(self, config, pars, sample_id, net, outsz, log=None, is_forward=False):

        self.setsz_ins  = net.get_inputs_setsz()
        self.setsz_outs = net.get_outputs_setsz()

        # initialize the general sample
        CSample.__init__(self, config, pars, sample_id, net, \
                         outsz, self.setsz_ins, self.setsz_outs, \
                         log=log, is_forward=is_forward)

        # precompute the global rebalance weights
        self._prepare_rebalance_weights()

    def _prepare_rebalance_weights(self):
        # rebalance weights
        self.wps = dict()
        self.wzs = dict()
        for key, lbl in self.lbls.iteritems():
            msk = self.msks[key]
            self.wps[key], self.wzs[key] = self._get_balance_weight( lbl,msk )

    def _binary_class(self, lbl):
        """
        Binary-Class Label Transformation

        Parameters
        ----------
        lbl : 4D array, label volume.

        Return
        ------
        ret : 4D array, two volume with opposite value
        """
        assert(lbl.shape[0] == 1)

        ret = np.empty((2,)+ lbl.shape[1:4], dtype= self.pars['dtype'])
        ret[0, :,:,:] = (lbl[0,:,:,:]>0).astype(self.pars['dtype'])
        ret[1:,  :,:,:] = 1 - ret[0, :,:,:]

        return ret

    def _rebalance_bdr(self, sublbl, submsk, wp, wz):
        assert sublbl.ndim==4 and sublbl.shape[0]==1

        weight = np.ones( sublbl.shape, dtype=self.pars['dtype'] )

        # recompute weight for patch rebalance
        if self.pars['rebalance_mode'] and 'patch' in self.pars['rebalance_mode']:
            wp, wz = self._get_balance_weight_v1( sublbl,submsk )

        if self.pars['rebalance_mode']:
            weight[0,:,:,:][sublbl[0,:,:,:]> 0] = wp
            weight[0,:,:,:][sublbl[0,:,:,:]==0] = wz

        return weight


    def get_random_sample(self):
        subimgs, sublbls, submsks = super(CBoundarySample, self).get_random_sample()

        # boudary map rebalance
        subwmsks = dict()
        for key, sublbl in sublbls.iteritems():
            submsk = submsks[key]
            subwmsks[key] = self._rebalance_bdr( sublbl, submsk, self.wps[key], self.wzs[key] )

        for key,sublbl in sublbls.iteritems():
            assert sublbl.ndim==3 or (sublbl.ndim==4 and sublbl.shape[0]==1)
            # binarize the true lable
            sublbls[key] = self._binary_class( sublbl )
            # duplicate the maskes
            submsks[key]  = np.tile(submsks[key], (2,1,1,1))
            subwmsks[key] = np.tile(subwmsks[key], (2,1,1,1))

        return subimgs, sublbls, submsks, subwmsks

class ConfigSampleOutput(object):
    '''Documentation coming soon...'''

    def __init__(self, pars, net, output_volume_shape3d):

        output_patch_shapes = net.get_outputs_setsz()

        self.output_volumes = {}
        for name, shape in output_patch_shapes.iteritems():

            num_volumes = shape[0]

            volume_shape = np.hstack((num_volumes, output_volume_shape3d)).astype('uint32')

            empty_bin = np.zeros(volume_shape, dtype=pars['dtype'])


            self.output_volumes[name] = CDataset(pars, empty_bin, shape[-3:], shape[-3:] )

    def set_next_patch(self, output):

        for name, data in output.iteritems():
            self.output_volumes[name].set_next_patch(data)

    def num_patches(self):

        patch_counts = {}

        for name, dataset in self.output_volumes.iteritems():
            patch_counts[name] = dataset.num_patches()

        return patch_counts

class CSamples(object):

    def __init__(self, config, pars, ids, net, outsz, log=None):
        """
        Samples Class - which represents a collection of data samples

        This can be useful when one needs to use multiple collections
        of data for training/testing, or as a generalized interface
        for single collections

        Parameters
        ----------
        config : python parser object, read the config file
        pars : parameters
        ids : set of sample ids
        net: network for which this samples object should be tailored
        """

        #Parameter object
        self.pars = pars

        self.samples = list()
        # probability of choosing this sample
        self.smp_prbs = list()
        # total number of candidate locations
        Nloc = 0.0
        for sid in ids:
            if 'bound' in pars['out_type']:
                sample = CBoundarySample(config, pars, sid, net, outsz, log)
            elif 'aff' in pars['out_type']:
                sample = CAffinitySample(config, pars, sid, net, outsz, log)
            elif 'semantic' in pars['out_type']:
                sample = CSemanticSample(config, pars, sid, net, outsz, log)
            else:
                raise NameError('invalid output type')
            self.samples.append( sample )
            self.smp_prbs.append(len(sample.locs))
            Nloc += len(sample.locs)
        # normalize the number of locations to probability
        for i,p in enumerate(self.smp_prbs):
            self.smp_prbs[i] = self.smp_prbs[i] / Nloc

        if self.pars['is_debug']:
            # save the candidate locations
            self._save_dataset
            self._save_candidate_locs()

    def _save_candidate_locs(self):
        for sample in self.samples:
            fname = '../testsuit/candidate_locs_{}.h5'.format(sample.sid)
            if os.path.exists( fname ):
                os.remove(fname)
            f = h5py.File( fname, 'w' )
            f.create_dataset('locs', data=sample.locs)
            f.close()

    def _save_dataset(self):
        from emirt.emio import imsave
        for sample in self.samples:
            # save sample images
            raw, lbl = sample.get_dataset()
            fname = '../testsuit/sample_{}_raw.h5'.format(sample.sid)
            if os.path.exists( fname ):
                os.remove( fname )
            imsave(raw, fname)
            fname = '../testsuit/sample_{}_lbl.h5'.format(sample.sid)
            if os.path.exists( fname ):
                os.remove( fname )
            imsave(lbl, fname )

    def get_random_sample(self):
        '''Fetches a random sample from a random CSample object'''
        # get the sequence of multinomial sequence. only one element is 1, others are 0
        sq = np.random.multinomial(1, self.smp_prbs, size=1)
        # get the index of non-zero element
        i = np.nonzero(sq)[1][0]
        return self.samples[i].get_random_sample()
