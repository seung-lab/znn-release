#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import emirt
import time
import ConfigParser
import cost_fn
import matplotlib.pylab as plt

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
    pars['dp_type']     = config.get('parameters', 'dp_type')

    pars['train_net']        = config.get('parameters', 'train_net')
    pars['train_range'] = parseIntSet( config.get('parameters',   'train_range') )
    pars['test_range']  = parseIntSet( config.get('parameters',   'test_range') )
    pars['eta']         = config.getfloat('parameters', 'eta')
    pars['anneal_factor']=config.getfloat('parameters', 'anneal_factor')
    pars['momentum']    = config.getfloat('parameters', 'momentum')
    pars['weight_decay']= config.getfloat('parameters', 'weight_decay')
    pars['train_outsz']       = np.asarray( [x for x in config.get('parameters', 'train_outsz').split(',') ], dtype=np.int64 )
    pars['is_data_aug'] = config.getboolean('parameters', 'is_data_aug')
    pars['is_rebalance']= config.getboolean('parameters', 'is_rebalance')
    pars['is_malis']    = config.getboolean('parameters', 'is_malis')
    pars['cost_fn_str'] = config.get('parameters', 'cost_fn')

    pars['Num_iter_per_show'] = config.getint('parameters', 'Num_iter_per_show')
    pars['Num_iter_per_test'] = config.getint('parameters', 'Num_iter_per_test')
    pars['Num_iter_per_save'] = config.getint('parameters', 'Num_iter_per_save')
    pars['Max_iter']    = config.getint('parameters', 'Max_iter')

    # forward parameters
    pars['forward_range'] = parseIntSet( config.get('parameters', 'forward_range') )
    pars['forward_net']   = config.get('parameters', 'forward_net')
    pars['forward_outsz'] = np.asarray( [x for x in config.get('parameters', 'forward_outsz').split(',') ], dtype=np.int64 )
    pars['output_prefix'] = config.get('parameters', 'output_prefix')

    # cost function
    if pars['cost_fn_str'] == "square_loss":
        pars['cost_fn'] = cost_fn.square_loss
    elif pars['cost_fn_str'] == "binomial_cross_entropy":
        pars['cost_fn'] = cost_fn.binomial_cross_entropy
    elif pars['cost_fn_str'] == "multinomial_cross_entropy":
        pars['cost_fn'] = cost_fn.multinomial_cross_entropy
    else:
        raise NameError('unknown type of cost function')

    #%% print parameters
    if pars['is_rebalance']:
        print "rebalance the gradients"
    if pars['is_malis']:
        print "using malis weight"
        
    #%% assert some options
    if pars['is_malis']:
        if 'aff' not in pars['dp_type']:
            raise NameError( 'malis weight should be used with affinity label type!' )
    return config, pars

class CSample:
    def _read_files(self, files):
        """
        read a list of tif files of original volume and lable

        Parameters
        ----------
        files : list of string, file names

        Return
        ------
        ret:  list of 3D array
        """
        ret = list()
        for fl in files:
            vol = emirt.emio.imread(fl).astype('float32')
            ret.append( vol )
        return ret

    def _lbl2aff( self ):
        """
        transform labels to affinity
        """
        assert( len(self.lbls)==1 )
        lbl = self.lbls[0]
        aff = np.zeros((3,)+lbl.shape, dtype='float32')
        aff[0,1:,:,:] = (lbl[1:,:,:] == lbl[:-1,:,:]) & (lbl[1:,:,:]>0)
        aff[1,:,1:,:] = (lbl[:,1:,:] == lbl[:,:-1,:]) & (lbl[:,1:,:]>0)
        aff[2,:,:,1:] = (lbl[:,:,1:] == lbl[:,:,:-1]) & (lbl[:,:,1:]>0)
        self.lbls = aff

    def _preprocess_vol(self, vol, pp_type):
        if 'standard2D' == pp_type:
            for z in xrange( vol.shape[0] ):
                vol[z,:,:] = (vol[z,:,:] - np.mean(vol[z,:,:])) / np.std(vol[z,:,:])
        elif 'standard3D' == pp_type:
            vol = (vol - np.mean(vol)) / np.std(vol)
        elif 'none' == pp_type:
            return vol
        else:
            raise NameError( 'invalid preprocessing type' )
        return vol

    def _preprocess(self):
        vols = self.vols
        self.vols = list()
        for vol, pp_type in zip(vols, self.pp_types):
            vol = self._preprocess_vol(vol, pp_type)
            self.vols.append( vol )

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

    def _auto_crop(self):
        """
        crop the list of volumes to make sure that volume sizes are the same.
        Note that this function was not tested yet!!
        """
        if len(self.vols) == 1:
            return
        # find minimum size
        splist = list()
        for vol in self.vols:
            splist.append( vol.shape )
        sz_min = min( splist )

        # crop every volume
        for k in xrange( len(self.vols) ):
            self.vols[k] = self._center_crop( self.vols[k], sz_min )
        return

    def _threshold_label(self):
        for k,lbl in enumerate( self.lbls ):
            self.lbls[k] = (lbl>0).astype('float32')
        
    """class of sample, similar with Dataset module of pylearn2"""
    def __init__(self, sample_id, config, pars):
        self.pars = pars
        dp_type = pars['dp_type']

        sec_name = "sample%d" % (sample_id,)
        fvols  = config.get(sec_name, 'fvols').split(',\n')
        self.vols = self._read_files( fvols )       
        
        if config.has_option( sec_name, 'flbls' ):
            flbls  = config.get(sec_name, 'flbls').split(',\n')
            self.lbls = self._read_files( flbls )
            
        self.pp_types = config.get(sec_name, 'pp_type').split(',')
         
        # preprocess the input volumes
        self._preprocess()

        if config.getboolean(sec_name, 'is_auto_crop'):
            self._auto_crop()

        if config.has_option( sec_name, 'flbls' ):
            if 'aff' in dp_type:
                self._lbl2aff()
            elif 'vol' in dp_type or 'boundary' in dp_type:
                # threshold the lable
                self._threshold_label()

    def _get_random_subvol(self, insz, outsz):
        """
        get random sample from training and labeling volumes

        Parameters
        ----------
        insz :  input size.
        outsz:  output size of network.

        Returns
        -------
        vol_ins  : input volume of network.
        vol_outs : label volume of network.
        """
        c = len(self.vols)
        self.vol_ins = np.empty(np.hstack((c,insz)), dtype='float32')
        dp_type = self.pars['dp_type']
        if 'vol' in dp_type or 'boundary' in dp_type:
            self.lbl_outs= np.empty(np.hstack((1,outsz)), dtype='float32')
        elif 'aff' in dp_type:
            self.lbl_outs= np.empty(np.hstack((3,outsz)), dtype='float32')
        # configure size
        half_in_sz  = insz.astype('uint32')  / 2
        half_out_sz = outsz.astype('uint32') / 2
       # margin consideration for even-sized input
       # margin_sz = (insz-1) / 2
        set_sz = self.vols[0].shape - insz + 1
        # get random location
        loc = np.zeros(3)
        loc[0] = np.random.randint(half_in_sz[0], half_in_sz[0] + set_sz[0])
        loc[1] = np.random.randint(half_in_sz[1], half_in_sz[1] + set_sz[1])
        loc[2] = np.random.randint(half_in_sz[2], half_in_sz[2] + set_sz[2])
        # extract volume
        for k, vol in enumerate(self.vols):
            self.vol_ins[k,:,:,:]  = vol[   loc[0]-half_in_sz[0]  : loc[0]-half_in_sz[0] + insz[0],\
                                            loc[1]-half_in_sz[1]  : loc[1]-half_in_sz[1] + insz[1],\
                                            loc[2]-half_in_sz[2]  : loc[2]-half_in_sz[2] + insz[2]]
        for k, lbl in enumerate( self.lbls ):
            if 'vol' in dp_type or 'boundary' in dp_type:
                self.lbl_outs[0,:,:,:] = lbl[   loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
                                                loc[1]-half_out_sz[1] : loc[1]-half_out_sz[1]+outsz[1],\
                                                loc[2]-half_out_sz[2] : loc[2]-half_out_sz[2]+outsz[2]]
            elif 'aff' in dp_type:
                self.lbl_outs[:,:,:,:] = lbl[:, loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
                                                loc[1]-half_out_sz[1] : loc[1]-half_out_sz[1]+outsz[1],\
                                                loc[2]-half_out_sz[2] : loc[2]-half_out_sz[2]+outsz[2]]
        return (self.vol_ins, self.lbl_outs)

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
        # transform every pair of input and label volume
        if rft[0]:
            # first flip and than transpose
            if rft[1]:
                data  = np.fliplr( data )
                if rft[2]:
                    data  = np.flipud( data )
                    if rft[3]:
                        data = data[::-1, :,:]
            if rft[4]:
                data = data.transpose(0,2,1)
        else:
            # first transpose, than flip
            if rft[4]:
                data = data.transpose(0,2,1)
            if rft[1]:
                data = np.fliplr( data )
                if rft[2]:
                    data = np.flipud( data )
                    if rft[3]:
                        data = data[::-1, :,:]
        return data

    def _data_aug(self, vols, lbls, dp_type ):
        """
        data augmentation, transform volumes randomly to enrich the training dataset.

        Parameters
        ----------
        vol : input volumes of network.
        lbl : label volumes of network.

        Returns
        -------
        vol : transformed input volumes of network.
        lbl : transformed label volumes.
        """
        # random flip and transpose: flip-transpose order, fliplr, flipud, flipz, transposeXY
        rft = (np.random.random(5)>0.5)
        for i in xrange(vols.shape[0]):
            vols[i,:,:,:] = self._data_aug_transform(vols[i,:,:,:], rft)
        for i in xrange(lbls.shape[0]):
            lbls[i,:,:,:] = self._data_aug_transform(lbls[i,:,:,:], rft)
        return (vols, lbls)
    def get_random_sample(self, insz, outsz):
        dp_type = self.pars['dp_type']
        vins, vouts = self._get_random_subvol( insz, outsz )
        if self.pars['is_data_aug']:
            vins, vouts = self._data_aug( vins, vouts, dp_type )
        return ( vins, vouts )

class CSamples:
    def __init__(self, ids, config, pars):
        """
        Parameters
        ----------
        ids : vector of sample ids

        Return
        ------

        """
        self.samples = list()
        self.pars = pars
        for sid in ids:
            sample = CSample(sid, config, pars)
            self.samples.append( sample )

    def get_random_sample(self, insz, outsz):
        i = np.random.randint( len(self.samples) )
        vins, vouts = self.samples[i].get_random_sample( insz, outsz)
        return (vins, vouts)
        
    def get_inputs(self, sid):
        return self.samples[sid].vols

    def volume_dump(self):
        '''Returns ALL contained volumes

        Used within forward pass'''

        vols = []
        for i in range(len(self.samples)):
            vols.extend(self.samples[i].vols)

        return vols


def inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                titr_list, terr_list, tcls_list, \
                eta, vol_ins, props, lbl_outs, grdts, tpars, \
                rb_weights=0, malis_weights=0):
    # time
    elapsed = time.time() - start
    print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s, learning rate: %.4f"\
            %(i, err, cls, elapsed, eta )
    # real time visualization
    plt.subplot(331),   plt.imshow(vol_ins[0,0,:,:],       interpolation='nearest', cmap='gray')
    plt.xlabel('input')
    plt.subplot(332),   plt.imshow(props[0,0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('inference')
    plt.subplot(333),   plt.imshow(lbl_outs[0,0,:,:], interpolation='nearest', cmap='gray')
    plt.xlabel('lable')
    plt.subplot(334),   plt.imshow(grdts[0,0,:,:],     interpolation='nearest', cmap='gray')
    plt.xlabel('gradient')


    plt.subplot(337)
    plt.plot(it_list,   err_list,   'b', label='train')
    plt.plot(titr_list, terr_list,  'r', label='test')
    plt.xlabel('iteration'), plt.ylabel('cost energy')
    plt.subplot(338)
    plt.plot(it_list, cls_list, 'b', titr_list, tcls_list, 'r')
    plt.xlabel('iteration'), plt.ylabel( 'classification error' )

    # reset time
    start = time.time()
    # reset err and cls
    err = 0
    cls = 0

    if tpars['is_rebalance']:
        plt.subplot(335),   plt.imshow(   rb_weights[1,0,:,:],interpolation='nearest', cmap='gray')
        plt.xlabel('rebalance weight')
    if tpars['is_malis']:
        plt.subplot(335),   plt.imshow(np.log(malis_weights[1,0,:,:]),interpolation='nearest', cmap='gray')
        plt.xlabel('malis weight (log)')
    plt.pause(1)
    return start, err, cls
