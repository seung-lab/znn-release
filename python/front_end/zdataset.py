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

class CDataset(object):

    def __init__(self, pars, data, outsz, setsz, mapsz=None):

        # main data
        self.data = data
        # field of view of whole network
        if mapsz is None:
            self.mapsz = setsz[-3:] - outsz[-3:] + 1
        else:
            self.mapsz = mapsz

        # Desired size of subvolumes returned by this instance
        self.patch_shape = np.asarray(setsz[-3:])
        if pars['is_debug']:
            print "patch shape: ", self.patch_shape

        # (see check_patch_bounds, or _calculate_patch_bounds)
        self.net_output_patch_shape = outsz
        #Actually calculating patch bounds can be (somewhat) expensive
        # so we'll only define this if the user tries to use patches
        self.patch_bounds = None
        self.patch_id = 0

        # calculate some attribute sizes for further computation
        self.calculate_sizes()

    def calculate_sizes(self):
        '''
        Adjusts the shape attributes to account for a change in the
        shape of the data array

        Currently used to account for boundary mirroring within subclasses
        '''

        self.volume_shape = np.asarray(self.data.shape[-3:])

        # center coordinate
        # -1 accounts for python indexing
        self.center = (self.volume_shape-1) / 2

        #Number of voxels with index lower than the center
        # within a subvolume (used within get_dev_range, and
        # get_sub_volume)
        self.patch_margin_low  = (self.patch_shape-1) / 2
        #Number of voxels with index higher than the center
        # within a subvolume (used within get_dev_range, and
        # get_sub_volume)
        self.patch_margin_high = self.patch_shape / 2

    def _check_patch_bounds(self):
        if self.patch_bounds is None:
            self._calculate_patch_bounds()

    def _calculate_patch_bounds(self, output_patch_shape=None, overwrite=True):
        '''
        Finds the bounds for each data patch given the input volume shape,
        the network mapsz, and the output patch shape

        Restricts calculation to 3d shape
        '''

        if output_patch_shape is None:
            output_patch_shape = self.net_output_patch_shape

        #3d Shape restriction
        output_patch_shape = output_patch_shape[-3:]

        #Decomposing into a similar problem for each axis
        z_bounds = self._patch_bounds_1d(self.volume_shape[0],
                        output_patch_shape[0], self.mapsz[0])
        y_bounds = self._patch_bounds_1d(self.volume_shape[1],
                        output_patch_shape[1], self.mapsz[1])
        x_bounds = self._patch_bounds_1d(self.volume_shape[2],
                        output_patch_shape[2], self.mapsz[2])

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

    def _patch_bounds_1d(self, vol_width, patch_width, mapsz_width):

        bounds = []

        beginning = 0
        ending = patch_width + mapsz_width - 1

        while ending < vol_width:

            bounds.append(
                ( beginning, ending )
                )

            beginning += patch_width
            ending += patch_width

        #last bound
        bounds.append(
            ( vol_width - (patch_width + mapsz_width - 1), vol_width)
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

    def output_volume_shape(self):
        '''
        Determines the full output volume shape for the network given
        the entire input volume
        '''
        return self.volume_shape - self.mapsz + 1

class ConfigImage(CDataset):
    """
    A class which represents a stack of images (up to 4 dimensions)

    In the 4-dimensional case, it can constrain the constituent 3d volumes
    to be the same size.

    The design of the class is focused around returning subvolumes of a
    particular size (patch_shape). It can accomplish this by specifying a deviation
    (in voxels) from the center.
    """

    def __init__(self, config, pars, sec_name, \
                 outsz, setsz, mapsz, is_forward=False):
        """
        Parameters
        ----------
        config: data specification file parser
        pars: dict, parameters
        sec_name: string, section name in data spec file
        setsz: array, data patch shape
        outsz: network output size
        is_forward: binary, whether this is forward or not
        """
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
        if arr.ndim==3:
            arr = arr.reshape( (1,) + arr.shape )

        # initialize the dataset
        CDataset.__init__(self, pars, arr, outsz, setsz, mapsz)


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
            vol = emirt.emio.imread(fl)
            if vol.ndim==4:
                # read the VAST output RGB images
                print "reading RGB label image: ", fl
                assert( vol.dtype=='uint8' and vol.shape[3]==3 )
                vol = vol.astype('uint32')
                vol = vol[:,:,:,0]*256*256 + vol[:,:,:,1]*256 + vol[:,:,:,2]
            vol = vol.astype(self.pars['dtype'])
            ret.append( vol )
        return ret

    def get_dev_range(self):
        """
        Subvolumes can be specified in terms of 'deviation' from the center voxel
        (see get_subvolume below)

        This function specifies the valid range of those deviations in terms of
        xyz coordinates
        """

        # Number of voxels within index lower than the center
        volume_margin_low  = (self.volume_shape - 1) / 2
        # Number of voxels within index higher than the center
        volume_margin_high = self.volume_shape / 2

        lower_bound  = -( volume_margin_low - self.patch_margin_low )
        upper_bound  =   volume_margin_high - self.patch_margin_high

        if self.pars['is_debug']:
            print "vlome margin low: ", volume_margin_low
            print "patch_margin_low: ", self.patch_margin_low
            print "deviation range:     ", lower_bound, "--", upper_bound

        return lower_bound, upper_bound

    def get_subvolume(self, dev, data=None):
        """
        Returns a 4d subvolume of the data volume, specified
        by deviation from the center voxel.

        Can also retrieve subvolume of a passed 4d array

        Parameters
        ----------
        dev : the deviation from the whole volume center

        Return
        -------
        subvol : the transformed sub volume.
        """

        if data is None:
            data = self.data

        loc = self.center + dev

        # extract volume
        subvol  = np.copy(data[ :,
            loc[0]-self.patch_margin_low[0]  : loc[0] + self.patch_margin_high[0]+1,\
            loc[1]-self.patch_margin_low[1]  : loc[1] + self.patch_margin_high[1]+1,\
            loc[2]-self.patch_margin_low[2]  : loc[2] + self.patch_margin_high[2]+1])
        return subvol

class ConfigInputImage(ConfigImage):
    '''
    Subclass of ConfigImage which represents the type of input data seen
    by ZNN neural networks
    '''

    def __init__(self, config, pars, sec_name, \
                 outsz, setsz, mapsz, is_forward=False ):
        ConfigImage.__init__(self, config, pars, sec_name, \
                             outsz, setsz, mapsz, is_forward=is_forward )

        # preprocessing
        pp_types = config.get(sec_name, 'pp_types').split(',')
        assert self.data.shape[0]==1 and self.data.ndim==4
        for c in xrange( self.data.shape[0] ):
            self.data[c,:,:,:] = self._preprocess(self.data[c,:,:,:], pp_types[c])

        if pars['is_bd_mirror']:
            if self.pars['is_debug']:
                print "data shape before mirror: ", self.data.shape
            self.data = utils.boundary_mirror(self.data, self.mapsz)
            #Modifying the deviation boundaries for the modified dataset
            self.calculate_sizes( )
            if self.pars['is_debug']:
                print "data shape after mirror: ", self.data.shape

    def _preprocess( self, vol3d , pp_type ):

        if 'standard2D' == pp_type:
            for z in xrange( vol3d.shape[0] ):
                vol3d[z,:,:] = (vol3d[z,:,:] - np.mean(vol3d[z,:,:])) / np.std(vol3d[z,:,:])
        elif 'standard3D' == pp_type:
            vol3d = (vol3d - np.mean(vol3d)) / np.std(vol3d)
        elif 'symetric_rescale' == pp_type:
            # rescale to -1,1
            vol3d -= vol3d.min()
            vol3d = vol3d / vol3d.max()
            vol3d = vol3d * 2 - 1
        elif 'none' in pp_type or "None" in pp_type:
            return vol3d
        else:
            raise NameError( 'invalid preprocessing type' )

        return vol3d

    def get_subvolume(self, dev, data=None):
        """
        Returns a 4d subvolume of the original, specified
        by deviation from the center voxel.

        Parameters
        ----------
        dev : the deviation from the whole volume center

        Return:
        -------
        subvol : the transformed sub volume.
        """
        subvol = super(ConfigInputImage, self).get_subvolume(dev, data=data)
        assert(subvol.ndim==4)
        return subvol

    def get_dataset(self):
        """
        return complete volume for examination
        """
        return self.data

class ConfigOutputLabel(ConfigImage):
    '''
    Subclass of CImage which represents output labels for
    ZNN neural networks

    Internally handles preprocessing of the data, and can
    contain masks for sparsely-labelled training
    '''

    def __init__(self, config, pars, sec_name, outsz, setsz, mapsz ):
        ConfigImage.__init__(self, config, pars, sec_name, \
                             outsz, setsz, mapsz=mapsz)

        # record and use parameters
        self.pars = pars

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
                assert(self.data.shape == self.msk.shape)

        # preprocessing
        self.pp_types = config.get(sec_name, 'pp_types').split(',')

    def get_subvolume(self, dev):
        """
        get sub volume for training.

        Parameter
        ---------
        dev : deviation from the desired subvolume center

        Return
        ------
        sublbl  : 4D array, ground truth label,
        submsk  : 4D array, label mask
        """

        sublbl = super(ConfigOutputLabel, self).get_subvolume(dev)
        assert sublbl.shape[0]==1

        if np.size(self.msk)>0:
            submsk = super(ConfigOutputLabel, self).get_subvolume(dev, data=self.msk)
        else:
            submsk = np.array([])

        return sublbl, submsk

    def get_dataset(self):
        """
        return the whole label for examination
        """
        return self.data

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
