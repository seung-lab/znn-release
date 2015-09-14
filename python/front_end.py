#!/usr/bin/env python
__doc__ = """

Front-End Interface for ZNNv4

Jingpeng Wu <jingpeng.wu@gmail.com>,
Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""
import numpy as np
import ConfigParser
import cost_fn
import matplotlib.pylab as plt
import pyznn
import sys
import emirt
from numba import autojit

def parseIntSet(nputstr=""):
    """
    Allows users to specify a comma-delimited list of number ranges as sample selections.
    Specifically, parses a string which should contain a comma-delimited list of 
    either numerical values (e.g. 3), or a dash-separated range (e.g. 4-5).

    If the ranges are redundant (e.g. 3, 3-5), only one copy of the selection will
    be added to the result.

    IGNORES ranges which don't fit the desired format (e.g. 3&5)

    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
    """

    selection = set()
    invalid = set()

    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]

    for i in tokens:
       try:
          # typically, tokens are plain old integers
          selection.add(int(i))
       except:

          # if not, then it might be a range
          try:
             token = [int(k.strip()) for k in i.split('-')]
             if len(token) > 1:
                token.sort()
                # we have items seperated by a dash
                # try to build a valid range
                first = token[0]
                last = token[len(token)-1]
                for x in range(first, last+1):
                   selection.add(x)
          except:
             # not an int and not a range...
             invalid.add(i)

    return selection

def parser( conf_fname ):
    '''
    Parses a configuration file into a dictionary of options using
    the ConfigParser module
    '''

    config = ConfigParser.ConfigParser()
    config.read( conf_fname )

    pars = dict()

    #GENERAL OPTIONS
    #Network specification filename
    pars['fnet_spec']   = config.get('parameters', 'fnet_spec')
    #Number of threads to use
    pars['num_threads'] = int( config.get('parameters', 'num_threads') )
    #Output layer data type (e.g. 'boundary','affinity')
    pars['out_dtype']     = config.get('parameters', 'out_dtype')

    #IO OPTIONS
    #Filename under which we save the network
    pars['train_save_net'] = config.get('parameters', 'train_save_net')
    #Network filename to load
    pars['train_load_net'] = config.get('parameters', 'train_load_net')

    #TRAINING OPTIONS
    #Samples to use for training
    pars['train_range'] = parseIntSet( config.get('parameters',   'train_range') )
    #Samples to use for cross-validation
    pars['test_range']  = parseIntSet( config.get('parameters',   'test_range') )
    #Learning Rate
    pars['eta']         = config.getfloat('parameters', 'eta')
    #Learning Rate Annealing Factor
    pars['anneal_factor']=config.getfloat('parameters', 'anneal_factor')
    #Momentum Constant
    pars['momentum']    = config.getfloat('parameters', 'momentum')
    #Weight Decay
    pars['weight_decay']= config.getfloat('parameters', 'weight_decay')
    #Training Output Patch Shape
    pars['train_outsz'] = np.asarray( [x for x in config.get('parameters', \
                                    'train_outsz').split(',') ], dtype=np.int64 )
    #Whether to optimize the convolution computation by layer
    # (FFT vs Direct Convolution)
    pars['is_optimize'] = config.getboolean('parameters', 'is_optimize')
    #Whether to use data augmentation
    pars['is_data_aug'] = config.getboolean('parameters', 'is_data_aug')
    #Whether to use boundary mirroring
    pars['is_bd_mirror']= config.getboolean('parameters', 'is_bd_mirror')
    #Whether to use rebalanced training
    pars['is_rebalance']= config.getboolean('parameters', 'is_rebalance')
    #Whether to use malis cost
    pars['is_malis']    = config.getboolean('parameters', 'is_malis')
    #Whether to display progress plots
    pars['is_visual']   = config.getboolean('parameters', 'is_visual')    
    
    #Which Cost Function to Use (as a string)
    pars['cost_fn_str'] = config.get('parameters', 'cost_fn')

    #DISPLAY OPTIONS
    #How often to show progress to the screen
    pars['Num_iter_per_show'] = config.getint('parameters', 'Num_iter_per_show')
    #How often to check cross-validation error
    pars['Num_iter_per_test'] = config.getint('parameters', 'Num_iter_per_test')
    #How many output patches should derive cross-validation error
    pars['test_num']    = config.getint( 'parameters', 'test_num' )
    #How often to save the network
    pars['Num_iter_per_save'] = config.getint('parameters', 'Num_iter_per_save')
    #Maximum training updates
    pars['Max_iter']    = config.getint('parameters', 'Max_iter')

    #FULL FORWARD PASS PARAMETERS
    #Which samples to use
    pars['forward_range'] = parseIntSet( config.get('parameters', 'forward_range') )
    #Which network file to load
    pars['forward_net']   = config.get('parameters', 'forward_net')
    #Output Patch Size
    pars['forward_outsz'] = np.asarray( [x for x in config.get('parameters', 'forward_outsz')\
                                        .split(',') ], dtype=np.int64 )
    #Prefix of the output files
    pars['output_prefix'] = config.get('parameters', 'output_prefix')

    #PROCESSING COST FUNCTION STRING
    if pars['cost_fn_str'] == "square_loss":
        pars['cost_fn'] = cost_fn.square_loss
    elif pars['cost_fn_str'] == "binomial_cross_entropy":
        pars['cost_fn'] = cost_fn.binomial_cross_entropy
    elif pars['cost_fn_str'] == "multinomial_cross_entropy":
        pars['cost_fn'] = cost_fn.multinomial_cross_entropy
    elif pars['cost_fn_str'] == "softmax_loss":
        pars['cost_fn'] = cost_fn.softmax_loss
    else:
        raise NameError('unknown type of cost function')

    #%% check the consistency of some options
    if pars['is_malis']:
        if 'aff' not in pars['out_dtype']:
            raise NameError( 'malis weight should be used with affinity label type!' )
    return config, pars

def init_network( params=None, train=True, output_patch_shape=None,
            network_specfile=None, num_threads=None, optimize=None ):
    '''
    Initializes a random network using the Boost Python interface and configuration
    file options. The function will define this network by a parameter object
    (as generated by the parse function), or by the specified options.

    If both a parameter object and any optional arguments are specified,
    the parameter object will form the default options, and those will be 
    overwritten by the other optional arguments
    '''
    #Need to specify either a params object, or all of the other optional args
    params_defined = params is not None

    #"ALL" optional args excludes train
    all_optional_args_defined = all([
        arg is not None for arg in 
        (output_patch_shape, network_specfile, num_threads, optimize)
        ])

    assert (params_defined or all_optional_args_defined)

    #Defining phase argument by train argument
    phase = int(not train)

    #If a params object exists, then those options are the default
    if params is not None:

        if train:
            _output_patch_shape = params['train_outsz']
        else:
            _output_patch_shape = params['forward_outsz']

        _network_specfile = params['fnet_spec']
        _num_threads = params['num_threads']
        _optimize = params['is_optimize']

    #Overwriting defaults with any other optional args
    if output_patch_shape is not None:
        _output_patch_shape = output_patch_shape
    if network_specfile is not None:
        _network_specfile = network_specfile
    if num_threads is not None:
        _num_threads = num_threads
    if optimize is not None:
        _optimize = optimize

    return pyznn.CNet(_network_specfile, _output_patch_shape, 
                    _num_threads, _optimize, phase)

class CImage:
    """
    A class which represents a stack of images (up to 4 dimensions)

    In the 4-dimensional case, it can constrain the constituent 3d volumes
    to be the same size.

    The design of the class is focused around returning subvolumes of a
    particular size (setsz). It can accomplish this by specifying a deviation
    (in voxels) from the center. The class also internally performs
    rotations and flips for data augmentation.
    """

    def __init__(self, config, pars, sec_name, setsz):

        #Parameter object (see parser above)
        self.pars = pars
        #Desired size of subvolumes returned by this instance
        self.setsz = setsz

        #Reading in data
        fnames = config.get(sec_name, 'fnames').split(',\n')
        arrlist = self._read_files( fnames );
        
        #Auto crop - constraining 3d vols to be the same size
        self._is_auto_crop = config.getboolean(sec_name, 'is_auto_crop')
        if self._is_auto_crop:
            arrlist = self._auto_crop( arrlist )

        #4d array of all data
        self.arr = np.asarray( arrlist, dtype='float32')
        #3d shape of a constituent volume
        self.sz = np.asarray( self.arr.shape[1:4] )
        
        #Computes center coordinate, picks lower-index priority center
        self.center = self._get_center()

        #Number of voxels with index lower than the center
        # within a subvolume (used within get_div_range, and
        # get_sub_volume)
        self.low_setsz  = (self.setsz-1)/2
        #Number of voxels with index higher than the center
        # within a subvolume (used within get_div_range, and
        # get_sub_volume)
        self.high_setsz = self.setsz / 2

        #Display some instance information
        print "image stack size:    ", self.arr.shape
        print "set size:            ", self.setsz
        print "center:              ", self.center
        return

    def get_div_range(self):
        """
        Subvolumes are specified in terms of 'deviation' from the center voxel

        This function specifies the valid range of those deviations in terms of
        xyz coordinates
        """

        #Number of voxels within index lower than the center
        low_sz  = (self.sz - 1) /2
        #Number of voxels within index higher than the center
        high_sz = self.sz/2

        low  = -( low_sz - self.low_setsz )
        high = high_sz - self.high_setsz

        print "deviation range:     ", low, "--", high

        return low, high

    def _get_center(self):
        '''
        Finds the index of the 3d center of the array
        
        Picks the lower-index voxel if there's no true "center"
        '''

        #-1 accounts for python indexing
        center = (self.sz-1)/2
        return center

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
            vol = emirt.emio.imread(fl).astype('float32')
            ret.append( vol )
        return ret

    def get_sub_volume(self, arr, div, rft=[]):
        """
        Returns a 4d subvolume of the original, specified
        by deviation from the center voxel. Performs data
        augmentation if specified by the rft argument

        Parameters
        ----------
        div : the deviation from the center
        rft : the random transformation rule.
        Return:
        -------
        subvol : the transformed sub volume.
        """

        # the center location
        loc = self.center + div

        # extract volume
        subvol  = arr[ :,   loc[0]-self.low_setsz[0]  : loc[0] + self.high_setsz[0]+1,\
                            loc[1]-self.low_setsz[1]  : loc[1] + self.high_setsz[1]+1,\
                            loc[2]-self.low_setsz[2]  : loc[2] + self.high_setsz[2]+1]
        # random transformation
        if self.pars['is_data_aug']:
            subvol = self._data_aug_transform(subvol, rft)
        return subvol

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

        if np.size(rft)==0:
            return data
        # transform every pair of input and label volume

        #z-reflection
        if rft[0]:
            data  = data[:, ::-1, :,    :]
        #y-reflection
        if rft[1]:
            data  = data[:, :,    ::-1, :]
        #x-reflection
        if rft[2]:
            data = data[:,  :,    :,    ::-1]
        #transpose
        if rft[3]:
            data = data.transpose(0,1,3,2)

        return data

class CInputImage(CImage):
    '''
    Subclass of CImage which represents the type of input data seen
    by ZNN neural networks 

    Internally preprocesses the data, and modifies the legal 
    deviation range for affinity data output.
    '''

    def __init__(self, config, pars, sec_name, setsz ):
        CImage.__init__(self, config, pars, sec_name, setsz )

        # preprocessing
        pp_types = config.get(sec_name, 'pp_types').split(',')
        for c in xrange( self.arr.shape[0] ):
            self.arr[c,:,:,:] = self._preprocess(self.arr[c,:,:,:], pp_types[c])

    def _preprocess( self, vol, pp_type):
        if 'standard2D' == pp_type:
            for z in xrange( vol.shape[0] ):
                vol[z,:,:] = (vol[z,:,:] - np.mean(vol[z,:,:])) / np.std(vol[z,:,:])
        elif 'standard3D' == pp_type:
            vol = (vol - np.mean(vol)) / np.std(vol)
        elif 'none' == pp_type or "None" in pp_type:
            return vol
        else:
            raise NameError( 'invalid preprocessing type' )
        return vol

    def get_subvol(self, div, rft):
        return self.get_sub_volume(self.arr, div, rft)

    def get_div_range(self):
        '''Override of the CImage implementation to account
        for affinity preprocessing'''

        low, high = super(CInputImage, self).get_div_range()

        if 'aff' in self.pars['out_dtype']:
            #Given affinity preprocessing (see _lbl2aff), valid affinity
            # values only exist for the later voxels, which can create
            # boundary issues
            low += 1

        return low, high

class COutputLabel(CImage):
    '''
    Subclass of CImage which represents output labels for
    ZNN neural networks 

    Internally handles preprocessing of the data, and can 
    contain masks for sparsely-labelled training
    '''

    def __init__(self, config, pars, sec_name, setsz):
        CImage.__init__(self, config, pars, sec_name, setsz)

        # Affinity preprocessing decreases the output
        # size by one voxel in each dimension, this counteracts
        # that effect
        if 'aff' in pars['out_dtype']:
            # increase the subvolume size for affinity
            self.setsz = self.setsz + 1

        # deal with mask
        self.msk = np.array([])
        if config.has_option(sec_name, 'fmasks'):
            fmasks = config.get(sec_name, 'fnames').split(',\n')
            msklist = self._read_files( fmasks )

            if self._is_auto_crop:
                msklist = self._auto_crop( msklist )

            self.msk = np.asarray( msklist )
            # mask 'preprocessing'
            self.msk = (self.msk>0).astype('float32')

            assert(self.arr.shape == self.msk.shape)   
            
        
        if pars['is_rebalance']:
            self._rebalance()
            
        # preprocessing
        self.pp_types = config.get(sec_name, 'pp_types').split(',')        
        self._preprocess()       

    def _preprocess( self ):
        """
        preprocess the 4D image stack.

        Parameters
        ----------
        arr : 3D array,
        """

        self.pp_types = config.get(sec_name, 'pp_types').split(',')

        assert(len(self.pp_types)==1)

        # loop through volumes
        for c, pp_type in enumerate(self.pp_types):
            if 'none' == pp_type or 'None'==pp_type:
                return
            elif 'binary_class' == pp_type:
                self.arr = self._binary_class(self.arr)
                self.msk = np.tile(self.msk, (2,1,1,1))
                return
            elif 'one_class' == pp_type:
                self.arr = (self.arr>0).astype('float32')
                return
            elif 'aff' in pp_type:
                # affinity preprocessing handled later
                # when fetching subvolumes (get_subvol)
                return
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

        ret = np.empty((2,)+ lbl.shape[1:4], dtype='float32')

        ret[0, :,:,:] = (lbl[0,:,:,:]>0).astype('float32')
        ret[1:,  :,:,:] = 1 - ret[0, :,:,:]

        return ret

    def get_subvol(self, div, rft):
        """
        get sub volume for training.

        Parameter
        ---------
        div : coordinate array, deviation from volume center.
        rft : binary vector, transformation rule

        Return
        ------
        arr : 4D array, could be affinity of binary class
        """
        sublbl = self.get_sub_volume(self.arr, div, rft)
        submsk = self.get_sub_volume(self.msk, div, rft)
        if 'aff' in self.pp_types[0]:
            # transform the output volumes to affinity array
            sublbl = self._lbl2aff( sublbl )
            # get the affinity mask
            submsk = self._msk2affmsk( submsk )
            if self.pars['is_rebalance']:
                # apply the rebalance
                submsk = self._rebalance_aff(sublbl, submsk)
        return sublbl, submsk
    
    def _rebalance_aff(self, lbl, msk):
        wts = np.zeros(lbl.shape, dtype='float32')
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
    
    @autojit
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
        ret = np.zeros((3, Z-1, Y-1, X-1), dtype='float32')
        
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
        lbl : 4D float32 array, label volume.

        Returns
        -------
        aff : 4D float32 array, affinity graph.
        """
        # the 3D volume number should be one
        assert( lbl.shape[0] == 1 )

        aff_size = np.asarray(lbl.shape)-1
        aff_size[0] = 3

        aff = np.zeros( tuple(aff_size) , dtype='float32')

        #x-affinity
        aff[0,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,:-1, 1:  ,1: ]) & (lbl[0,1:,1:,1:]>0)
        #y-affinity
        aff[1,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , :-1 ,1: ]) & (lbl[0,1:,1:,1:]>0)
        #z-affinity
        aff[2,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , 1:  ,:-1]) & (lbl[0,1:,1:,1:]>0)

        return aff

    def _get_balance_weight( self, arr ):
        # number of nonzero elements
        num_nz = float( np.count_nonzero(arr) )
        # total number of elements
        num = float( np.size(arr) )

        # weight of positive and zero
        wp = 0.5 * num / num_nz
        wz = 0.5 * num / (num - num_nz)
        return wp, wz
    def _rebalance( self ):
        """
        get rebalance tree_size of gradient.
        make the nonboundary and boundary region have same contribution of training.
        """
        if 'aff' in self.pp_types[0]:
            zlbl = (self.arr[0,1:,1:,1:] != self.arr[0, :-1, 1:,  1:])
            ylbl = (self.arr[0,1:,1:,1:] != self.arr[0, 1:,  :-1, 1:])
            xlbl = (self.arr[0,1:,1:,1:] != self.arr[0, 1:,  1:,  :-1])
            self.zwp, self.zwz = self._get_balance_weight(zlbl)
            self.ywp, self.ywz = self._get_balance_weight(ylbl)
            self.xwp, self.xwz = self._get_balance_weight(xlbl)
        else:
            # positive is non-boundary, zero is boundary
            wnb, wb = self._get_balance_weight(self.arr)
            # give value
            weight = np.empty( self.arr.shape, dtype='float32' )
            weight[self.arr>0]  = wnb
            weight[self.arr==0] = wb
    
            if np.size(self.msk)==0:
                self.msk = weight
            else:
                self.msk = self.msk * weight

class CSample:
    """
    Sample Class, which represents a pair of input and output volume structures
    (as CInputImage and COutputImage respectively) 

    Allows simple interface for procuring matched random samples from all volume
    structures at once

    Designed to be similar with Dataset module of pylearn2
    """
    def __init__(self, config, pars, sample_id, net):

        # Parameter object (dict)
        self.pars = pars

        #Extracting layer info from the network
        info_in  = net.get_inputs()
        info_out = net.get_outputs()

        # Name of the sample within the configuration file
        sec_name = "sample%d" % sample_id

        # init deviation range
        # we need to consolidate this over all input and output volumes
        self.div_high = np.array([sys.maxsize, sys.maxsize, sys.maxsize])
        self.div_low  = np.array([-sys.maxint-1, -sys.maxint-1, -sys.maxint-1])

        # Loading input images
        self.inputs = dict()
        for name,setsz in info_in.iteritems():

            #Finding the section of the config file
            imid = config.getint(sec_name, name)
            imsec_name = "image%d" % (imid,)
            
            self.inputs[name] = CInputImage(  config, pars, imsec_name, setsz[1:4] )
            low, high = self.inputs[name].get_div_range()

            # Deviation bookkeeping
            self.div_high = np.minimum( self.div_high, high )
            self.div_low  = np.maximum( self.div_low , low  )

        # define output images
        self.outputs = dict()
        for name, setsz in info_out.iteritems():

            #Finding the section of the config file
            imid = config.getint(sec_name, name)
            imsec_name = "label%d" % (imid,)

            self.outputs[name] = COutputLabel( config, pars, imsec_name, setsz[1:4])
            low, high = self.outputs[name].get_div_range()

            # Deviation bookkeeping
            self.div_high = np.minimum( self.div_high, high )
            self.div_low  = np.maximum( self.div_low , low  )
        # find the candidate central locations of sample
        

    def get_random_sample(self):
        '''Fetches a matching random sample from all input and output volumes'''

        # random transformation roll
        rft = (np.random.rand(4)>0.5)

        # random deviation from the volume center
        div = np.empty(3)
        div[0] = np.random.randint(self.div_low[0], self.div_high[0])
        div[1] = np.random.randint(self.div_low[1], self.div_high[1])
        div[2] = np.random.randint(self.div_low[2], self.div_high[2])

        # get input and output 4D sub arrays
        inputs = dict()
        for name, img in self.inputs.iteritems():
            inputs[name] = img.get_subvol(div, rft)

        outputs = dict()
        msks = dict()
        for name, lbl in self.outputs.iteritems():
            outputs[name], msks[name] = lbl.get_subvol(div, rft)

        return ( inputs, outputs, msks )

class CSamples:
    def __init__(self, config, pars, ids, net):
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

        #Information about the input and output layers
        info_in  = net.get_inputs()
        info_out = net.get_outputs()
        
        self.samples = list()
        for sid in ids:
            sample = CSample(config, pars, sid, net)
            self.samples.append( sample )

    def get_random_sample(self):
        '''Fetches a random sample from a random CSample object'''
        i = np.random.randint( len(self.samples) )
        return self.samples[i].get_random_sample()

    def get_inputs(self, sid):
        return self.samples[sid].get_input()

def inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                titr_list, terr_list, tcls_list, \
                eta, vol_ins, props, lbl_outs, grdts, pars):
    '''
    Plots a display of training information to the screen
    '''
    name_in, vol  = vol_ins.popitem()
    name_p,  prop = props.popitem()
    name_l,  lbl  = lbl_outs.popitem()
    name_g,  grdt = grdts.popitem()


    # real time visualization
    plt.subplot(241),   plt.imshow(vol[0,0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('input')
    plt.subplot(242),   plt.imshow(prop[0,0,:,:],   interpolation='nearest', cmap='gray')
    plt.xlabel('inference')
    plt.subplot(243),   plt.imshow(lbl[0,0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('label')
    plt.subplot(244),   plt.imshow(grdt[0,0,:,:],   interpolation='nearest', cmap='gray')
    plt.xlabel('gradient')


    plt.subplot(245)
    plt.plot(it_list,   err_list,   'b', label='train')
    plt.plot(titr_list, terr_list,  'r', label='test')
    plt.xlabel('iteration'), plt.ylabel('cost energy')
    plt.subplot(247)
    plt.plot(it_list, cls_list, 'b', titr_list, tcls_list, 'r')
    plt.xlabel('iteration'), plt.ylabel( 'classification error' )

    plt.pause(1.5)
    return
