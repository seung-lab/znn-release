#!/usr/bin/env python
__doc__ = """

Dataset Class Interface (CSamples)

Jingpeng Wu <jingpeng.wu@gmail.com>,
Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""

import sys

import numpy as np
from numba import autojit

import emirt
import utils

class ZNN_Dataset(object):

    def __init__(self, data, data_patch_shape, net_output_patch_shape):

        self.data = data
        #Desired size of subvolumes returned by this instance
        self.patch_shape = np.asarray(data_patch_shape[-3:])

        self.volume_shape = np.asarray(self.data.shape[-3:])
        # network field of view (wrt THIS dataset)
        self.fov = data_patch_shape[-3:] - net_output_patch_shape + 1        
        # center coordinate
        #-1 accounts for python indexing
        self.center = (self.volume_shape-1) / 2

        #Number of voxels with index lower than the center
        # within a subvolume (used within get_dev_range, and
        # get_sub_volume)
        self.patch_margin_low  = (self.patch_shape-1) / 2
        #Number of voxels with index higher than the center
        # within a subvolume (used within get_dev_range, and
        # get_sub_volume)
        self.patch_margin_high = self.patch_shape / 2

        #Display some instance information (...meh)
        # print "image stack size:    ", self.arr.shape
        # print "set size:            ", self.setsz
        # print "center:              ", self.center

        #Used in calculating patch bounds if using patch functionality
        # (see check_patch_bounds, or _calculate_patch_bounds)
        self.net_output_patch_shape = net_output_patch_shape
        #Actually calculating patch bounds can be (somewhat) expensive
        # so we'll only define this if the user tries to use patches
        self.patch_bounds = None
        self.patch_id = 0

    def get_dev_range(self):
        """
        Subvolumes can be specified in terms of 'deviation' from the center voxel
        (see get_subvolume below)

        This function specifies the valid range of those deviations in terms of
        xyz coordinates
        """

        #Number of voxels within index lower than the center
        volume_margin_low  = (self.volume_shape - 1) / 2
        #Number of voxels within index higher than the center
        volume_margin_high = self.volume_shape / 2

        lower_bound  = -( volume_margin_low - self.patch_margin_low )
        upper_bound  =   volume_margin_high - self.patch_margin_high

        print "deviation range:     ", lower_bound, "--", upper_bound

        return lower_bound, upper_bound

    def get_subvolume(self, dev, rft=None, data=None):
        """
        Returns a 4d subvolume of the data volume, specified
        by deviation from the center voxel. Performs data
        augmentation if specified by the rft argument.

        Can also retrieve subvolume of a passed 4d array

        Parameters
        ----------
        dev : the deviation from the whole volume center
        rft : the random transformation rule.
        
        Return
        -------
        subvol : the transformed sub volume.
        """

        if data is None:
            data = self.data 

        loc = self.center + dev

        # extract volume
        subvol  = data[ :,   
            loc[0]-self.patch_margin_low[0]  : loc[0] + self.patch_margin_high[0]+1,\
            loc[1]-self.patch_margin_low[1]  : loc[1] + self.patch_margin_high[1]+1,\
            loc[2]-self.patch_margin_low[2]  : loc[2] + self.patch_margin_high[2]+1]

        if rft is not None:
            subvol = utils.data_aug_transform(subvol, rft)
            
        return subvol

    def _check_patch_bounds(self):

        if self.patch_bounds is None:
            print "Calculating patch bounds"
            self._calculate_patch_bounds()
            print "Done"

    def _calculate_patch_bounds(self, output_patch_shape=None, overwrite=True):
        '''
        Finds the bounds for each data patch given the input volume shape,
        the network fov, and the output patch shape

        Restricts calculation to 3d shape
        '''

        if output_patch_shape is None:
            output_patch_shape = self.net_output_patch_shape

        #3d Shape restriction
        output_patch_shape = output_patch_shape[-3:]

        #Decomposing into a similar problem for each axis
        z_bounds = self._patch_bounds_1d(self.volume_shape[0], 
                        output_patch_shape[0], self.fov[0])
        y_bounds = self._patch_bounds_1d(self.volume_shape[1], 
                        output_patch_shape[1], self.fov[1])
        x_bounds = self._patch_bounds_1d(self.volume_shape[2], 
                        output_patch_shape[2], self.fov[2])

        #And then recombining the subproblems
        bounds = []
        for z in z_bounds:
            for y in y_bounds:
                for x in x_bounds:
                    bounds.append(
                            (
                            #beginning for each axis
                            (z[0],y[0],x[0]),
                            #ending for each axis
                            (z[1],y[1],x[1])
                            )
                        )

        if overwrite:
            self.patch_bounds = bounds
            return
        else:
            return bounds

    def _patch_bounds_1d(self, vol_width, patch_width, fov_width):

        bounds = []

        beginning = 0
        ending = patch_width + fov_width - 1

        while ending < vol_width:

            bounds.append(
                ( beginning, ending )
                )

            beginning += patch_width
            ending += patch_width

        #last bound
        bounds.append(
            ( vol_width - (patch_width + fov_width - 1), vol_width)
            )

        return bounds

    def get_patch(self, patch_id):

        #Checking whether patch bounds are defined
        self._check_patch_bounds()

        patch_beginnings = self.patch_bounds[patch_id][0]
        patch_ends = self.patch_bounds[patch_id][1]

        return self.data[ :,
                    patch_beginnings[0]:patch_ends[0],
                    patch_beginnings[1]:patch_ends[1],
                    patch_beginnings[2]:patch_ends[2]]

    def get_next_patch(self):
        
        #Checking whether patch bounds are defined
        self._check_patch_bounds()

        patch = self.get_patch(self.patch_id)
        self.patch_id += 1

        return patch

    def set_patch(self, data, patch_id):
        
        #Checking whether patch bounds are defined
        self._check_patch_bounds()

        patch_beginnings = self.patch_bounds[patch_id][0]
        patch_ends = self.patch_bounds[patch_id][1]

        self.data[ :,
                patch_beginnings[0]:patch_ends[0],
                patch_beginnings[1]:patch_ends[1],
                patch_beginnings[2]:patch_ends[2]] = data

    def set_next_patch(self, data):

        #Checking whether patch bounds are defined
        self._check_patch_bounds()

        self.set_patch(data, self.patch_id)
        self.patch_id += 1

    def num_patches(self):
        #Checking whether patch bounds are defined
        self._check_patch_bounds()

        return len(self.patch_bounds)

    def has_next_patch(self):
        #Checking whether patch bounds are defined
        self._check_patch_bounds()

        return self.patch_id < self.num_patches()

    def reset(self):
        self.patch_id = 0

class ConfigImage(ZNN_Dataset):
    """
    A class which represents a stack of images (up to 4 dimensions)

    In the 4-dimensional case, it can constrain the constituent 3d volumes
    to be the same size.

    The design of the class is focused around returning subvolumes of a
    particular size (patch_shape). It can accomplish this by specifying a deviation
    (in voxels) from the center. The class also internally performs
    rotations and flips for data augmentation.
    """

    def __init__(self, config, pars, sec_name, setsz, outsz):

        #Parameter object (see parser above)
        self.pars = pars

        #Reading in data
        fnames = config.get(sec_name, 'fnames').split(',\n')
        arrlist = self._read_files( fnames );
        
        #Auto crop - constraining 3d vols to be the same size
        self._is_auto_crop = config.getboolean(sec_name, 'is_auto_crop')
        if self._is_auto_crop:
            arrlist = self._auto_crop( arrlist )

        #4d array of all data
        arr = np.asarray( arrlist, dtype=pars['dtype'])
        ZNN_Dataset.__init__(self, arr, setsz, outsz)

    def _recalculate_sizes(self, net_output_patch_shape):
        '''
        Adjusts the shape attributes to account for a change in the
        shape of the data array

        Currently used to account for boundary mirroring within subclasses
        '''

        self.volume_shape = np.asarray(self.data.shape[-3:])

        self.center = (self.volume_shape-1) / 2
        self.fov = self.patch_shape[-3:] - net_output_patch_shape + 1        

        #Number of voxels with index lower than the center
        # within a subvolume (used within get_dev_range, and
        # get_sub_volume)
        self.patch_margin_low  = (self.patch_shape-1) / 2
        #Number of voxels with index higher than the center
        # within a subvolume (used within get_dev_range, and
        # get_sub_volume)
        self.patch_margin_high = self.patch_shape / 2

    def _center_crop(self, vol, shape):
        """
        Crops the passed volume from the center

        Parameters
        ----------
        vol : the array to be croped
        shape : the croped shape

        Returns
        -------
        vol : the croped volume
        """

        sz1 = np.asarray( vol.shape )
        sz2 = np.asarray( shape )
        # offset of both sides
        off1 = (sz1 - sz2+1)/2
        off2 = (sz1 - sz2)/2

        return vol[ off1[0]:-off2[0],\
                    off1[1]:-off2[1],\
                    off1[2]:-off2[2]]

    def _auto_crop(self, arrs):
        """
        crop the list of volumes to make sure that volume sizes are the same.

        Note that this function was not tested yet!!
        """

        if len(arrs) == 1:
            return arrs

        # find minimum size
        splist = list()
        for arr in arrs:
            splist.append( arr.shape )
        sz_min = min( splist )

        # crop every volume
        ret = list()
        for arr in arrs:
            ret.append( self._center_crop( arr, sz_min ) )
        return ret

    def _read_files(self, files):
        """
        read a list of tif files

        Parameters
        ----------
        files : list of string, file names

        Return
        ------
        ret:  list of 3D array, could be different size
        """
        ret = list()
        for fl in files:
            vol = emirt.emio.imread(fl).astype(self.pars['dtype'])
            if vol.dtype=='uint8' and vol.shape[3]==3:
                # read the VAST output RGB images
                vol = vol.astpye('uint32')
                vol = vol[:,:,:,0]*256*256 + vol[:,:,:,1]*256 + vol[:,:,:,2]
            ret.append( vol )
        return ret

    def get_subvolume(self, dev, rft=[], data=None):
        """
        Returns a 4d subvolume of the original, specified
        by deviation from the center voxel. Performs data
        augmentation if specified by the rft argument

        Parameters
        ----------
        dev : the deviation from the whole volume center
        rft : the random transformation rule.
        Return:
        -------
        subvol : the transformed sub volume.
        """
        if not self.pars['is_data_aug']:
            rft = None

        return super(ConfigImage, self).get_subvolume(dev, rft=rft, data=data)

class ConfigInputImage(ConfigImage):
    '''
    Subclass of ConfigImage which represents the type of input data seen
    by ZNN neural networks 

    Internally preprocesses the data, and modifies the legal 
    deviation range for affinity data output.
    '''

    def __init__(self, config, pars, sec_name, setsz, outsz ):
        ConfigImage.__init__(self, config, pars, sec_name, setsz, outsz )
        
        if pars['is_bd_mirror']:
            self.data = utils.boundary_mirror(self.data, self.fov)
            #Modifying the deviation boundaries for the modified dataset
            self._recalculate_sizes( outsz )

        # preprocessing
        pp_types = config.get(sec_name, 'pp_types').split(',')
        for c in xrange( self.data.shape[0] ):
            self.data[c,:,:,:] = self._preprocess(self.data[c,:,:,:], pp_types[c])

    def _preprocess( self, vol3d , pp_type ):

        if 'standard2D' == pp_type:
            for z in xrange( vol3d.shape[0] ):
                vol3d[z,:,:] = (vol3d[z,:,:] - np.mean(vol3d[z,:,:])) / np.std(vol3d[z,:,:])

        elif 'standard3D' == pp_type:
            vol3d = (vol3d - np.mean(vol3d)) / np.std(vol3d)

        elif 'none' == pp_type or "None" in pp_type:
            return vol3d

        else:
            raise NameError( 'invalid preprocessing type' )

        return vol3d

    def get_dev_range(self):
        '''Override of the CImage implementation to account
        for affinity preprocessing'''

        low, high = super(ConfigInputImage, self).get_dev_range()

        if 'aff' in self.pars['out_type']:
            #Given affinity preprocessing (see _lbl2aff), valid affinity
            # values only exist for the later voxels, which can create
            # boundary issues
            low += 1

        return low, high

class ConfigOutputLabel(ConfigImage):
    '''
    Subclass of CImage which represents output labels for
    ZNN neural networks 

    Internally handles preprocessing of the data, and can 
    contain masks for sparsely-labelled training
    '''

    def __init__(self, config, pars, sec_name, setsz, outsz):
        ConfigImage.__init__(self, config, pars, sec_name, setsz, outsz)
        
        # Affinity preprocessing decreases the output
        # size by one voxel in each dimension, this counteracts
        # that effect
        if 'aff' in pars['out_type']:
            # increase the subvolume size for affinity
            self.patch_shape += 1
            self._recalculate_sizes( outsz )        
        
        if pars['is_bd_mirror']:
            self.data = utils.boundary_mirror(self.data, self.fov)
            #Modifying the deviation boundaries for the modified dataset
            self._recalculate_sizes( outsz)

        # deal with mask
        self.msk = np.array([])
        if config.has_option(sec_name, 'fmasks'):
            fmasks = config.get(sec_name, 'fmasks').split(',\n')
            if fmasks[0]:
                msklist = self._read_files( fmasks )
                if self._is_auto_crop:
                    msklist = self._auto_crop( msklist )
                self.msk = np.asarray( msklist )
                # mask 'preprocessing'
                self.msk = (self.msk>0).astype(self.data.dtype)
                if pars['is_bd_mirror']:
                    self.msk = utils.boundary_mirror(self.msk, self.fov)
                assert(self.data.shape == self.msk.shape)   
                
        # preprocessing
        self.pp_types = config.get(sec_name, 'pp_types').split(',')        
        self._preprocess()       
        
        if pars['is_rebalance']:
            self._rebalance()

    def _preprocess( self ):
        """
        preprocess the 4D image stack.

        Parameters
        ----------
        arr : 3D array,
        """

        assert(len(self.pp_types)==1)

        # loop through volumes
        for c, pp_type in enumerate(self.pp_types):
            if 'none' == pp_type or 'None'==pp_type:
                pass

            elif 'binary_class' == pp_type:
                self.data = self._binary_class(self.data)
                self.msk = np.tile(self.msk, (2,1,1,1))

            elif 'one_class' == pp_type:
                self.data = (self.data>0).astype(self.pars['dtype'])

            elif 'aff' in pp_type:
                # affinity preprocessing handled later
                # when fetching subvolumes (get_subvolume)
                pass

            else:
                raise NameError( 'invalid preprocessing type' )

        return

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
        
        # fill the contacting segments with boundaries
        lbl[0,:,:,:] = utils.fill_boundary( lbl[0,:,:,:] ) 
        
        ret = np.empty((2,)+ lbl.shape[1:4], dtype= self.pars['dtype'])
        ret[0, :,:,:] = (lbl[0,:,:,:]>0).astype(self.pars['dtype'])
        ret[1:,  :,:,:] = 1 - ret[0, :,:,:]

        return ret

    def get_subvolume(self, dev, rft=[]):
        """
        get sub volume for training.

        Parameter
        ---------
        dev : deviation from the desired subvolume center
        rft : binary vector, transformation rule

        Return
        ------
        sublbl : 4D array, could be affinity or binary class
        submsk : 4D array, mask could contain rebalance weight
        """
        
        sublbl = super(ConfigOutputLabel, self).get_subvolume(dev, rft)

        if np.size(self.msk)>0:
            submsk = super(ConfigOutputLabel, self).get_subvolume(dev, rft, data=self.msk)
        else:
            submsk = np.array([])

        if 'aff' in self.pp_types[0]:
            # transform the output volumes to affinity array
            sublbl = self._lbl2aff( sublbl )

            # get the affinity mask
            if np.size(self.msk)>0:
                #?? not sure what this is doing yet
                submsk = self._msk2affmsk( submsk )

                if self.pars['is_rebalance']:
                    # apply the rebalancing
                    submsk = self._rebalance_aff(sublbl, submsk)
                    
        return sublbl, submsk
    
    def _rebalance_aff(self, lbl, msk):
        wts = np.zeros(lbl.shape, dtype=self.pars['dtype'])
        wts[0,:,:,:][lbl[0,:,:,:] >0] = self.zwp
        wts[1,:,:,:][lbl[1,:,:,:] >0] = self.ywp
        wts[2,:,:,:][lbl[2,:,:,:] >0] = self.xwp
        
        wts[0,:,:,:][lbl[0,:,:,:]==0] = self.zwz  
        wts[1,:,:,:][lbl[1,:,:,:]==0] = self.ywz
        wts[2,:,:,:][lbl[2,:,:,:]==0] = self.xwz
        if np.size(msk)==0:
            return wts
        else:
            return msk*wts
    
    @autojit(nopython=True)
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
        
        for z in xrange(Z-1):
            for y in xrange(Y-1):
                for x in xrange(X-1):
                    if msk[0,z,y,x]>0:
                        if msk[0,z+1,y,x]>0:
                            ret[0,z,y,x] = 1
                        if msk[0,z,y+1,x]>0:
                            ret[1,z,y,x] = 1
                        if msk[0,z,y,x+1]>0:
                            ret[2,z,y,x] = 1
        return ret
        
    def _lbl2aff( self, lbl ):
        """
        transform labels to affinity.

        Parameters
        ----------
        lbl : 4D float array, label volume.

        Returns
        -------
        aff : 4D float array, affinity graph.
        """
        # the 3D volume number should be one
        assert( lbl.shape[0] == 1 )

        aff_size = np.asarray(lbl.shape)-1
        aff_size[0] = 3

        if np.any( aff_size<0 ):
            print "oh, affinity size is less than 0!"
        aff = np.zeros( tuple(aff_size) , dtype=self.pars['dtype'])

        #x-affinity
        aff[0,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,:-1, 1:  ,1: ]) & (lbl[0,1:,1:,1:]>0)
        #y-affinity
        aff[1,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , :-1 ,1: ]) & (lbl[0,1:,1:,1:]>0)
        #z-affinity
        aff[2,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , 1:  ,:-1]) & (lbl[0,1:,1:,1:]>0)

        return aff

    def _get_balance_weight( self, arr ):
        # number of nonzero elements
        pn = float( np.count_nonzero(arr) )
        # total number of elements
        num = float( np.size(arr) )
        zn = num - pn

        # weight of positive and zero
        wp = 0.5 * num / pn
        wz = 0.5 * num / zn
        print "rebalance positive weight: ", wp
        print "rebalance zero     weight: ", wz
        return wp, wz

    def _rebalance( self ):
        """
        get rebalance tree_size of gradient.
        make the nonboundary and boundary region have same contribution of training.
        """
        if 'aff' in self.pp_types[0]:
            zlbl = (self.data[0,1:,1:,1:] == self.data[0, :-1, 1:,  1:])
            ylbl = (self.data[0,1:,1:,1:] == self.data[0, 1:,  :-1, 1:])
            xlbl = (self.data[0,1:,1:,1:] == self.data[0, 1:,  1:,  :-1])
            self.zwp, self.zwz = self._get_balance_weight(zlbl)
            self.ywp, self.ywz = self._get_balance_weight(ylbl)
            self.xwp, self.xwz = self._get_balance_weight(xlbl)
        else:
            weight = np.empty( self.data.shape, dtype=self.data.dtype )
            for c in xrange( self.data.shape[0] ):
                # positive is non-boundary, zero is boundary
                wp, wz = self._get_balance_weight(self.data[c,:,:,:])
                # give value
                weight[c,:,:,:][self.data[c,:,:,:]> 0] = wp
                weight[c,:,:,:][self.data[c,:,:,:]==0] = wz

            if np.size(self.msk)==0:
                self.msk = weight
            else:
                self.msk = self.msk * weight

    def get_candidate_loc( self, low, high ):
        """
        find the candidate location of subvolume
        
        Parameters
        ----------
        low  : vector with length of 3, low value of deviation range
        high : vector with length of 3, high value of deviation range
        
        Returns:
        --------
        ret : a tuple, the coordinate of nonzero elements,
              format is the same with return of numpy.nonzero.
        """
        if np.size(self.msk) == 0:
            mask = np.ones(self.data.shape[1:4], dtype=self.data.dtype)
        else:
            mask = np.copy(self.msk[0,:,:,:])
        # erase outside region of deviation range.
        ct = self.center
        mask[:ct[0]+low[0], :, : ] = 0
        mask[:, :ct[1]+low[1], : ] = 0
        mask[:, :, :ct[2]+low[2] ] = 0
        
        mask[ct[0]+high[0]+1:, :, :] = 0
        mask[:, ct[1]+high[1]+1:, :] = 0
        mask[:, :, ct[2]+high[2]+1:] = 0

        locs = np.nonzero(mask)

        if np.size(locs[0])==0:
            raise NameError('no candidate location!')

        return locs

class ConfigSample(object):
    """
    Sample Class, which represents a pair of input and output volume structures
    (as CInputImage and COutputImage respectively) 

    Allows simple interface for procuring matched random samples from all volume
    structures at once

    Designed to be similar with Dataset module of pylearn2
    """
    def __init__(self, config, pars, sample_id, net, outsz):

        # Parameter object (dict)
        self.pars = pars

        #Extracting layer info from the network
        setsz_ins  = net.get_inputs_setsz()
        setsz_outs = net.get_outputs_setsz()

        # Name of the sample within the configuration file
        sec_name = "sample%d" % sample_id

        # init deviation range
        # we need to consolidate this over all input and output volumes
        dev_high = np.array([sys.maxsize, sys.maxsize, sys.maxsize])
        dev_low  = np.array([-sys.maxint-1, -sys.maxint-1, -sys.maxint-1])

        # Loading input images
        print "\ncreate input image class..."
        self.inputs = dict()
        for name,setsz in setsz_ins.iteritems():

            #Finding the section of the config file
            imid = config.getint(sec_name, name)
            imsec_name = "image%d" % (imid,)
            
            self.inputs[name] = ConfigInputImage(  config, pars, imsec_name, setsz, outsz )
            low, high = self.inputs[name].get_dev_range()

            # Deviation bookkeeping
            dev_high = np.minimum( dev_high, high )
            dev_low  = np.maximum( dev_low , low  )

        # define output images
        print "\ncreate label image class..."
        self.outputs = dict()
        for name, setsz in setsz_outs.iteritems():

            #Finding the section of the config file
            imid = config.getint(sec_name, name)
            imsec_name = "label%d" % (imid,)

            self.outputs[name] = ConfigOutputLabel( config, pars, imsec_name, setsz, outsz)
            low, high = self.outputs[name].get_dev_range()

            # Deviation bookkeeping
            dev_high = np.minimum( dev_high, high )
            dev_low  = np.maximum( dev_low , low  )

        # find the candidate central locations of sample
        lbl = self.outputs.values()[0]
        self.locs = lbl.get_candidate_loc( dev_low, dev_high )
        

    def get_random_sample(self):
        '''Fetches a matching random sample from all input and output volumes'''

        # random transformation roll
        rft = (np.random.rand(4)>0.5)

        # random deviation
        ind = np.random.randint( np.size(self.locs[0]) )
        loc = np.empty( 3, dtype=np.uint32 )
        loc[0] = self.locs[0][ind]
        loc[1] = self.locs[1][ind]
        loc[2] = self.locs[2][ind]
        dev = self.outputs.values()[0].center - loc
        
        # get input and output 4D sub arrays
        inputs = dict()
        for name, img in self.inputs.iteritems():
            inputs[name] = img.get_subvolume(dev, rft)

        outputs = dict()
        msks = dict()
        for name, lbl in self.outputs.iteritems():
            outputs[name], msks[name] = lbl.get_subvolume(dev, rft)

        return ( inputs, outputs, msks )

class CSamples:
    def __init__(self, config, pars, ids, net, outsz):
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
        for sid in ids:
            sample = ConfigSample(config, pars, sid, net, outsz)
            self.samples.append( sample )

    def get_random_sample(self):
        '''Fetches a random sample from a random CSample object'''
        i = np.random.randint( len(self.samples) )
        return self.samples[i].get_random_sample()

