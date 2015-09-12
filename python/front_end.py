#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import time
import ConfigParser
import cost_fn
import matplotlib.pylab as plt
import sys
import emirt

def parseIntSet(nputstr=""):
    "http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html"
    selection = set()
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
       try:
          # typically tokens are plain old integers
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
    config = ConfigParser.ConfigParser()
    config.read( conf_fname )

    # general, train and forward
    pars = dict()

    pars['fnet_spec']   = config.get('parameters', 'fnet_spec')
    pars['num_threads'] = int( config.get('parameters', 'num_threads') )
    pars['out_dtype']     = config.get('parameters', 'out_dtype')

    pars['train_save_net'] = config.get('parameters', 'train_save_net')
    pars['train_load_net'] = config.get('parameters', 'train_load_net')
    pars['train_range'] = parseIntSet( config.get('parameters',   'train_range') )
    pars['test_range']  = parseIntSet( config.get('parameters',   'test_range') )
    pars['eta']         = config.getfloat('parameters', 'eta')
    pars['anneal_factor']=config.getfloat('parameters', 'anneal_factor')
    pars['momentum']    = config.getfloat('parameters', 'momentum')
    pars['weight_decay']= config.getfloat('parameters', 'weight_decay')
    pars['train_outsz'] = np.asarray( [x for x in config.get('parameters', \
                                    'train_outsz').split(',') ], dtype=np.int64 )
    
    pars['is_optimize'] = config.getboolean('parameters', 'is_optimize')
    pars['is_data_aug'] = config.getboolean('parameters', 'is_data_aug')
    pars['is_bd_mirror']= config.getboolean('parameters', 'is_bd_mirror')
    pars['is_rebalance']= config.getboolean('parameters', 'is_rebalance')
    pars['is_malis']    = config.getboolean('parameters', 'is_malis')
    pars['is_visual']   = config.getboolean('parameters', 'is_visual')    
    
    pars['cost_fn_str'] = config.get('parameters', 'cost_fn')

    pars['Num_iter_per_show'] = config.getint('parameters', 'Num_iter_per_show')
    pars['Num_iter_per_test'] = config.getint('parameters', 'Num_iter_per_test')
    pars['test_num']    = config.getint( 'parameters', 'test_num' )
    pars['Num_iter_per_save'] = config.getint('parameters', 'Num_iter_per_save')
    pars['Max_iter']    = config.getint('parameters', 'Max_iter')

    # forward parameters
    pars['forward_range'] = parseIntSet( config.get('parameters', 'forward_range') )
    pars['forward_net']   = config.get('parameters', 'forward_net')
    pars['forward_outsz'] = np.asarray( [x for x in config.get('parameters', 'forward_outsz')\
                                        .split(',') ], dtype=np.int64 )
    pars['output_prefix'] = config.get('parameters', 'output_prefix')

    # cost function
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

class CImage:
    """
    the image stacks.
    """
    def __init__(self, config, pars, sec_name, setsz):
        self.setsz = setsz
        self.pars = pars
        fnames = config.get(sec_name, 'fnames').split(',\n')
        arrlist = self._read_files( fnames );
        # auto crop
        self._is_auto_crop = config.getboolean(sec_name, 'is_auto_crop')
        if self._is_auto_crop:
            arrlist = self._auto_crop( arrlist )
        self.arr = np.asarray( arrlist, dtype='float32')
        self.sz = np.asarray( self.arr.shape[1:4] )
        
        # compute center coordinate
        self.center = self._get_center()
        
        # deal with affinity
        if 'aff' in pars['out_dtype']:
            # increase the subvolume size for affinity
            self.setsz = self.setsz + 1
        self.low_setsz  = (self.setsz-1)/2
        self.high_setsz = self.setsz / 2
        # show some information
        print "image stack size:    ", self.arr.shape
        print "set size:            ", self.setsz
        print "center:              ", self.center
        return
    
    def get_div_range(self):
        """
        get the range of diviation
        """
        low_setsz_div  = (self.setsz-1)  /2
        high_setsz_div = (self.setsz)    /2
        low_sz  = (self.sz - 1) /2
        high_sz = self.sz/2
        low  = -( low_sz - low_setsz_div )
        high = high_sz - high_setsz_div
        print "deviation range:     ", low, "--", high
        return low, high
    
    def _get_center(self):
        sz = np.asarray( self.arr.shape[1:4] );
        center = (sz-1)/2
        return center

    def _center_crop(self, vol, shape):
        """
        crop the volume from the center

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
        read a list of tif files of original volume and lable

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
        get sub volume.

        Parameters
        ----------
        div : the diviation from the center
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
        rft : transform rule

        Returns
        -------
        data : the transformed array
        """
        if np.size(rft)==0:
            return data
        # transform every pair of input and label volume
        if rft[0]:
            data  = data[:, ::-1, :,    :]
        if rft[1]:
            data  = data[:, :,    ::-1, :]
        if rft[2]:
            data = data[:,  :,    :,    ::-1]
        if rft[3]:
            data = data.transpose(0,1,3,2)
        return data

class CInputImage(CImage):
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
        arr = self.get_sub_volume(self.arr, div, rft)
        if 'aff' in self.pars['out_dtype']:
            # shrink the volume
            arr = arr[:,1:,1:,1:]
        return arr

class COutputLabel(CImage):
    def __init__(self, config, pars, sec_name, setsz):
        CImage.__init__(self, config, pars, sec_name, setsz)

        # deal with mask
        self.msk = []
        if config.has_option(sec_name, 'fmasks'):
            fmasks = config.get(sec_name, 'fnames').split(',\n')
            msklist = self._read_files( fmasks )
            if self._is_auto_crop:
                msklist = self._auto_crop( msklist )
            self.msk = np.asarray( msklist )
            self.msk = (self.msk>0).astype('float32')
            assert(self.arr.shape == self.msk.shape)   
            
        # preprocessing
        self._preprocess(config, sec_name)
        
        if pars['is_rebalance']:
            self._rebalance()

    def _preprocess( self, config, sec_name):
        """
        preprocess the 4D image stack.
        
        Parameters
        ----------
        arr : 3D array,
        """
        self.pp_types = config.get(sec_name, 'pp_types').split(',')
        assert(len(self.pp_types)==1)
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
                return
            else:
                raise NameError( 'invalid preprocessing type' )
        return
    
    def _binary_class(self, lbl):
        """
        transform label to binary class
        
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
            # shrink and replicate mask
            submsk = submsk[:,1:,1:,1:]
            submsk = np.tile(submsk, (3,1,1,1))
            
        return sublbl, submsk
    
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
        aff[0,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,:-1, 1:  ,1: ]) & (lbl[0,1:,1:,1:]>0)
        aff[1,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , :-1 ,1: ]) & (lbl[0,1:,1:,1:]>0)
        aff[2,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , 1:  ,:-1]) & (lbl[0,1:,1:,1:]>0)
        return aff

    def _rebalance( self ):
        """
        get rebalance tree_size of gradient.
        make the nonboundary and boundary region have same contribution of training.
        """
        # number of nonzero elements
        num_nz = float( np.count_nonzero(self.arr) )
        # total number of elements
        num = float( np.size(self.arr) )

        # weight of non-boundary and boundary
        wnb = 0.5 * num / num_nz
        wb  = 0.5 * num / (num - num_nz)

        # give value
        weight = np.empty( self.arr.shape, dtype='float32' )
        weight[self.arr>0]  = wnb
        weight[self.arr==0] = wb
    
        if np.size(self.msk)==0:
            self.msk = weight
        else:
            self.msk = self.msk * weight            

class CSample:
    """class of sample, similar with Dataset module of pylearn2"""
    def __init__(self, config, pars, sample_id, info_in, info_out):
        self.pars = pars
        sec_name = "sample%d" % (sample_id,)
        
        # deviation range
        self.div_high = np.array([sys.maxsize, sys.maxsize, sys.maxsize])
        self.div_low  = np.array([-sys.maxint-1, -sys.maxint-1, -sys.maxint-1])
        # input
        self.inputs = dict()
        for name,setsz in info_in.iteritems():
            imid = config.getint(sec_name, name)
            imsec_name = "image%d" % (imid,)
            self.inputs[name] = CInputImage(  config, pars, imsec_name, setsz[1:4] )
            low, high = self.inputs[name].get_div_range()
            self.div_high = np.minimum( self.div_high, high )
            self.div_low  = np.maximum( self.div_low , low  )
        # output
        self.outputs = dict()
        for name, setsz in info_out.iteritems():
            imid = config.getint(sec_name, name)
            imsec_name = "label%d" % (imid,)
            self.outputs[name] = COutputLabel( config, pars, imsec_name, setsz[1:4])
            low, high = self.outputs[name].get_div_range()
            self.div_high = np.minimum( self.div_high, high )
            self.div_low  = np.maximum( self.div_low , low  )
        
    def get_random_sample(self):
        # random transformation rull
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
    def __init__(self, config, pars, ids, info_in, info_out):
        """
        Parameters
        ----------
        config : python parser object, read the config file
        pars : parameters
        ids : vector of sample ids
        info_in  : dict, mapping of input  layer name and size
        info_out : dict, mapping of output layer name and size
        """
        self.samples = list()
        self.pars = pars
        for sid in ids:
            sample = CSample(config, pars, sid, info_in, info_out)
            self.samples.append( sample )

    def get_random_sample(self):
        i = np.random.randint( len(self.samples) )
        return self.samples[i].get_random_sample()

    def get_inputs(self, sid):
        return self.samples[sid].get_input()

    def volume_dump(self):
        '''Returns ALL contained volumes

        Used within forward pass'''

        vols = []
        for i in range(len(self.samples)):
            vols.extend(self.samples[i].vols)

        return vols


def inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                titr_list, terr_list, tcls_list, \
                eta, vol_ins, props, lbl_outs, grdts, pars):
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
    plt.xlabel('lable')
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
