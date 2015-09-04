#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import time
import ConfigParser
import cost_fn
import matplotlib.pylab as plt
import utils

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
    pars['train_outsz'] = np.asarray( [x for x in config.get('parameters', 'train_outsz').split(',') ], dtype=np.int64 )
    pars['is_data_aug'] = config.getboolean('parameters', 'is_data_aug')
    pars['is_rebalance']= config.getboolean('parameters', 'is_rebalance')
    pars['is_malis']    = config.getboolean('parameters', 'is_malis')
    pars['cost_fn_str'] = config.get('parameters', 'cost_fn')

    pars['Num_iter_per_show'] = config.getint('parameters', 'Num_iter_per_show')
    pars['Num_iter_per_test'] = config.getint('parameters', 'Num_iter_per_test')
    pars['test_num']    = config.getint( 'parameters', 'test_num' )
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
       
    #%% check the consistency of some options
    if pars['is_malis']:
        if 'aff' not in pars['out_dtype']:
            raise NameError( 'malis weight should be used with affinity label type!' )
    return config, pars

class CSample:
    """class of sample, similar with Dataset module of pylearn2"""
    def __init__(self, sample_id, config, pars):
        self.pars = pars
        out_dtype = pars['out_dtype']

        sec_name = "sample%d" % (sample_id,)
        fvols  = config.get(sec_name, 'fvols').split(',\n')
        self.vols = utils.read_files( fvols )       
        
        self.lbls=[]
        if config.has_option( sec_name, 'flbls' ) and config.get(sec_name, 'flbls'):
            flbls  = config.get(sec_name, 'flbls').split(',\n')
            self.lbls = utils.read_files( flbls )
        self.msks=[]
        if config.has_option( sec_name, 'fmsks' ) and config.get(sec_name, 'fmsks'):
            fmsks  = config.get(sec_name, 'fmsks').split(',\n')
            self.msks = utils.read_files( fmsks )
            self.msks = utils.binarize( self.msks, dtype='float32' )
            
        # rebalance
        if pars['is_rebalance']:
            weights = self._rebalance( self.lbls )
            if self.msks:
                self.msks = utils.loa_mul(self.msks, weights)
            else:
                self.msks = weights
        
        # preprocess the input volumes
        pp_types = config.get(sec_name, 'pp_type').split(',')
        self.vols = utils.preprocess( self.vols, pp_types)
        
        # crop the surrounding region to fit the smallest size 
        if config.getboolean(sec_name, 'is_auto_crop'):
            self.vols = utils.auto_crop( self.vols )
        
        # process the label data
        if config.has_option( sec_name, 'flbls' ) and \
                ('vol' in out_dtype or 'boundary' in out_dtype):
            self.lbls = utils.binarize( self.lbls )
                
    def _rebalance( self, lbls ):
        """
        get rebalance tree_size of gradient.
        make the nonboundary and boundary region have same contribution of training.
        """
        weights = list()
        for k,lbl in enumerate(lbls):
            # number of nonzero elements
            num_nz = float( np.count_nonzero(lbl) )
            # total number of elements
            num = float( np.size(lbl) )
        
            # weight of non-boundary and boundary
            wnb = 0.5 * num / num_nz
            wb  = 0.5 * num / (num - num_nz)
        
            # give value
            weight = np.empty( lbl.shape, dtype='float32' )
            weight[lbl>0]  = wnb
            weight[lbl==0] = wb
            weights.append( weight )
        return weights

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
        out_dtype = self.pars['out_dtype']
        self.lbl_outs= np.empty(np.hstack((1,outsz)), dtype='float32')

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
            self.lbl_outs[0,:,:,:] = lbl[   loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
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
            data  = np.fliplr( data )
        if rft[1]:
            data  = np.flipud( data )
        if rft[2]:
            data = data[::-1, :,:]
        if rft[3]:
            data = data.transpose(0,2,1)
        return data

    def _data_aug(self, vols, lbls ):
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
        rft = (np.random.random(4)>0.5)
        for i in xrange(vols.shape[0]):
            vols[i,:,:,:] = self._data_aug_transform(vols[i,:,:,:], rft)
        for i in xrange(lbls.shape[0]):
            lbls[i,:,:,:] = self._data_aug_transform(lbls[i,:,:,:], rft)
        return (vols, lbls)
    
    def _lbl2aff( self, vins, lbl ):
        """
        transform labels to affinity
        """
        aff_size = np.asarray(lbl.shape)-1
        aff_size[0] = 3
        aff = np.zeros( tuple(aff_size) , dtype='float32')
        aff[0,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,:-1, 1:  ,1: ]) & (lbl[0,1:,1:,1:]>0)
        aff[1,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , :-1 ,1: ]) & (lbl[0,1:,1:,1:]>0)
        aff[2,:,:,:] = (lbl[0,1:,1:,1:] == lbl[0,1: , 1:  ,:-1]) & (lbl[0,1:,1:,1:]>0)
        
        # shrink the input volume
        vins = vins[:,1:,1:,1:]
        return vins, aff
    
    def get_random_sample(self, insz, outsz):
        out_dtype = self.pars['out_dtype']
        if 'vol' in out_dtype or 'boundary' in out_dtype:
            vins, vouts = self._get_random_subvol( insz, outsz )
            if self.pars['is_data_aug']:
                vins, vouts = self._data_aug( vins, vouts )
        elif 'aff' in out_dtype:
            vins, vouts = self._get_random_subvol( insz+1, outsz+1 )
            if self.pars['is_data_aug']:
                vins, vouts = self._data_aug( vins, vouts )
            vins, vouts = self._lbl2aff( vins, vouts )   
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
