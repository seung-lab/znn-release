__doc__ = """

Sample Class Interface (CSamples)

Sample Class, which represents a pair of input and output volume structures(as CImage and CLabel respectively)

Allows simple interface for procuring matched random samples from all volume structures at once

Jingpeng Wu <jingpeng.wu@gmail.com>,
Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""

import sys
import numpy as np
import emirt
import zutils
from zdataset import *
import os
from copy import deepcopy

class CSample(object):
    def __init__(self, dspec, pars, name, net, \
                 outsz, setsz_ins=None, setsz_outs=None,\
                 log=None, is_forward=False):

        # Parameter object (dict)
        self.pars = pars

        # initialize the datasets
        self.ins = dict()
        self.outs = dict()

        # initialize the patches
        self.subimgs = dict()
        self.sublbls = dict()
        self.submsks  = dict()

        # Name of the sample within the configuration file
        # Also used for logging
        self.name = name

        # temporary layer names
        if is_forward and setsz_ins is None and setsz_outs is None:
            print "forward pass, get setsz from network"
            self.smp_setsz_ins  = net.get_inputs_setsz()
            self.smp_setsz_outs = net.get_outputs_setsz()
        else:
            # keep the original setsz for network
            print "setsz ins: {}".format( setsz_ins )
            print "setsz_outs: {}".format( setsz_outs )
            self.net_setsz_ins  = deepcopy( setsz_ins )
            self.net_setsz_outs = deepcopy( setsz_outs )
            # increase setsz by jitter size, will shrink in data augmentation
            # only change the xy coordinate, keep the z the same size
            self.smp_setsz_ins  = deepcopy( setsz_ins )
            self.smp_setsz_outs = deepcopy( setsz_outs )
            for k, v in self.smp_setsz_ins.iteritems():
                self.smp_setsz_ins[k][2:] = v[2:] + pars['jitter_size']
            for k, v in self.smp_setsz_outs.iteritems():
                self.smp_setsz_outs[k][2:] = v[2:] + pars['jitter_size']
        # field of view
        self.fov = np.asarray(net.get_fov(), dtype='uint32')

        # Loading input images
        print "\ncreate input image class..."
        for name,setsz_in in self.smp_setsz_ins.iteritems():
            # section name of image
            imsec = dspec[self.name][name]
            self.ins[name] = CImage( dspec, pars, imsec, \
                                     outsz, setsz_in, self.fov, is_forward=is_forward )

        if not is_forward:
            print "\ncreate label image class..."
            for name,setsz_out in self.smp_setsz_outs.iteritems():
                #Allowing for users to abstain from specifying labels
                if not (dspec.has_key(self.name) and dspec[self.name].has_key(name)):
                    continue
                #Finding the section of the config file
                imsec = dspec[self.name][name]
                self.outs[name] = CLabel( dspec, pars, imsec, \
                                          outsz, setsz_out, self.fov)
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

        for name,setsz in self.smp_setsz_ins.iteritems():
            low, high = self.ins[name].get_dev_range()
            # Deviation bookkeeping
            dev_high = np.minimum( dev_high, high )
            dev_low  = np.maximum( dev_low , low  )

        # define output images
        for name, setsz in self.smp_setsz_outs.iteritems():
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

    def _data_aug_transform(self, data, rft):
        """
        transform data according to a rule

        Parameters
        ----------
        data : 3D numpy array need to be transformed
        rft : transform rule, specified as an array of bool
            [z-reflection,
            y-reflection,
            x-reflection,
            xy transpose]

        Returns
        -------
        data : the transformed array
        """

        if np.size(data)==0 or np.size(rft)==0:
            return data

        #z-reflection
        if rft[0]:
            data  = data[:, ::-1, :,    :]
        #y-reflection
        if rft[1]:
            data  = data[:, :,    ::-1, :]
        #x-reflection
        if rft[2]:
            data = data[:,  :,    :,    ::-1]
        # transpose in XY
        if rft[3]:
            data = data.transpose(0,1,3,2)

        return data

    def _jitter_crop(self, img, z, offsets,  retsz):
        """
        simulate jittering
        Parameters:
        img: 4D numpy array, image stack
        z: int, the section index to separate image stack
        offsets: the offsets of sub image stacks
        retsz: int vector, the array size of returned image stack

        Return:
        img: 3D numpy array, jittered and cropped image stack
        """
        ret = np.empty(retsz, dtype=img.dtype)
        print "image size: {}".format(img.shape)
        print "z: {}".format(z)
        print "offsets: {}".format(offsets)
        print "return size: {}".format( retsz )
        ret[:,0:z,:,:] = img[:, 0:z, offsets[0]:offsets[0]+retsz[2], offsets[1]:offsets[1]+retsz[3] ]
        ret[:,z:,:,:]  = img[:, z:,  offsets[2]:offsets[2]+retsz[2], offsets[3]:offsets[3]+retsz[3] ]
        return ret
    def _data_aug(self):
        """
        data augmentation
        """
        # random transformation roll
        if self.pars['is_data_aug']:
            # random transformation
            rft = (np.random.rand(4)>0.5)
            if self.pars['jitter_size']<=0:
                for key, subimg in self.subimgs.iteritems():
                    self.subimgs[key] = self._data_aug_transform(subimg,      rft )
                for key, sublbl in self.sublbls.iteritems():
                    submsk = self.submsks[key]
                    self.sublbls[key] = self._data_aug_transform(sublbl, rft )
                    self.submsks[key] = self._data_aug_transform(submsk, rft )
                return
            # jitter code
            # the starting section in the output image stack
            outjz = np.random.randint( self.smp_setsz_outs.values()[0][1] )
            # input offset by ofv
            inoffset = (self.smp_setsz_ins.values()[0][1] - self.smp_setsz_outs.values()[0][1] + 1) /2
            injz = outjz + inoffset
            # the XY offset of upper and lower image stacks
            offsets = np.random.randint( self.pars['jitter_size'], size=4 )

            for key, subimg in self.subimgs.iteritems():
                # jitter crop
                subimg = self._jitter_crop(subimg, injz, offsets, self.net_setsz_ins[key])
                # spatial transformation
                self.subimgs[key] = self._data_aug_transform(subimg, rft )

            for key, sublbl in self.sublbls.iteritems():
                submsk = self.submsks[key]
                # jitter crop
                sublbl = self._jitter_crop(sublbl, outjz, offsets, self.net_setsz_outs[key])
                submsk = self._jitter_crop(submsk, outjz, offsets, self.net_setsz_outs[key])
                # spatial transformation
                self.sublbls[key]  = self._data_aug_transform(sublbl, rft )
                self.submsks[key]  = self._data_aug_transform(submsk, rft )

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
        for key, img in self.ins.iteritems():
            self.subimgs[key] = self.ins[key].get_subvolume(dev)
        for key, lbl in self.outs.iteritems():
            self.sublbls[key], self.submsks[key] = self.outs[key].get_subvolume(dev)

        # data augmentation
        self._data_aug()
        # make sure that the input image is continuous in memory
        # the C++ core can not deal with numpy view
        self.subimgs = zutils.make_continuous(self.subimgs)
        return ( self.subimgs, self.sublbls, self.submsks )

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
            log_line1 = self.name
            log_line2 = "subvolume: [{},{},{}] requested".format(dev[0],dev[1],dev[2])
            zutils.write_to_log(self.log, log_line1)
            zutils.write_to_log(self.log, log_line2)

class CAffinitySample(CSample):
    """
    sample for affinity training
    """

    def __init__(self, dspec, pars, sample_id, net, outsz, log=None, is_forward=False):
        self.smp_setsz_ins  = net.get_inputs_setsz()
        self.smp_setsz_outs = net.get_outputs_setsz()

        if not is_forward:
            # increase the shape by 1 for affinity sample
            # this will be shrinked later
            print "increase set size..."
            for key, setsz in self.smp_setsz_ins.iteritems():
                self.smp_setsz_ins[key][-3:] += 1
            for key, setsz in self.smp_setsz_outs.iteritems():
                self.smp_setsz_outs[key][-3:] += 1

        # initialize the general sample
        CSample.__init__(self, dspec, pars, sample_id, net, \
                         outsz, setsz_ins = self.smp_setsz_ins, setsz_outs = self.smp_setsz_outs, \
                         log=log, is_forward=is_forward)

        # precompute the global rebalance weights
        self.taffs = dict()
        self.tmsks = dict()
        for k, out in self.outs.iteritems():
            self.taffs[k] = self._seg2aff( out.get_lbl() )
            self.tmsks[k] = self._msk2affmsk( out.get_msk() )

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
    def __init__(self, dspec, pars, sample_id, net, outsz, log=None, is_forward=False):

        self.smp_setsz_ins  = net.get_inputs_setsz()
        self.smp_setsz_outs = net.get_outputs_setsz()

        # initialize the general sample
        CSample.__init__(self, dspec, pars, sample_id, \
                        net, outsz, self.smp_setsz_ins, \
                        self.smp_setsz_outs, log=log, is_forward=is_forward)

        # precompute the global rebalance weights
        self._prepare_rebalance_weights()

    def _prepare_rebalance_weights(self):
        # rebalance weights
        self.wps = dict()
        self.wzs = dict()
        for key, out in self.outs.iteritems():
            self.wps[key], self.wzs[key] = self._get_balance_weight( out.get_lbl(),out.get_msk() )

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
            subwmsk = self._rebalance_bdr( sublbl, submsk, self.wps[key], self.wzs[key] )
            subwmsks[key] = subwmsk

            assert sublbl.ndim==3 or (sublbl.ndim==4 and sublbl.shape[0]==1)
            # binarize the true lable
            sublbls[key] = self._binary_class( sublbl )
            # duplicate the masks
            submsks[key]  = np.tile(submsk, (2,1,1,1))
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

    def __init__(self, dspec, pars, ids, net, outsz, log=None):
        """
        Samples Class - which represents a collection of data samples

        This can be useful when one needs to use multiple collections
        of data for training/testing, or as a generalized interface
        for single collections

        Parameters
        ----------
        dspec :  dict, the dataset specifications
        pars : dict, parameters
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
            # sample name section
            sample_name = "sample{}".format(sid)
            if 'bound' in pars['out_type']:
                sample = CBoundarySample(dspec, pars, sample_name, net, outsz, log)
            elif 'aff' in pars['out_type']:
                sample = CAffinitySample(dspec, pars, sample_name, net, outsz, log)
            elif 'semantic' in pars['out_type']:
                sample = CSemanticSample(dspec, pars, sample_name, net, outsz, log)
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
            self._save_dataset()
            self._save_candidate_locs()

    def _save_candidate_locs(self):
        for sample in self.samples:
            fname = '../testsuit/candidate_locs_{}.h5'.format(sample.name)
            if os.path.exists( fname ):
                os.remove(fname)
            import h5py
            f = h5py.File( fname, 'w' )
            f.create_dataset('locs', data=sample.locs)
            f.close()

    def _save_dataset(self):
        from emirt.emio import imsave
        for sample in self.samples:
            # save sample images
            raw, lbl = sample.get_dataset()
            fname = '../testsuit/{}_raw.h5'.format(sample.name)
            if os.path.exists( fname ):
                os.remove( fname )
            imsave(raw, fname)
            fname = '../testsuit/{}_lbl.h5'.format(sample.name)
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
