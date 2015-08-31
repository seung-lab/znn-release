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
import pyznn
from emirt import volume_util

def parser( conf_fname ):
    config = ConfigParser.ConfigParser()
    config.read( conf_fname )
    
    # general, train and forward
    gpars = dict()    
    tpars = dict()
    fpars = dict()
    
    gpars['fnet_spec']   = config.get('general', 'fnet_spec')
    gpars['fnet']        = config.get('general', 'fnet')
    gpars['num_threads'] = int( config.get('general', 'num_threads') )
    gpars['dp_type']     = config.get('general', 'dp_type')
    
    tpars['ftrns']       = config.get('train', 'ftrns').split(',\n')
    tpars['flbls']       = config.get('train', 'flbls').split(',\n')
    tpars['ftsts']       = config.get('train', 'ftsts').split(',\n')
    tpars['ftlbs']       = config.get('train', 'ftlbs').split(',\n')
    tpars['eta']         = config.getfloat('train', 'eta') 
    tpars['anneal_factor']=config.getfloat('train', 'anneal_factor')
    tpars['momentum']    = config.getfloat('train', 'momentum') 
    tpars['weight_decay']= config.getfloat('train', 'weight_decay')
    tpars['outsz']       = np.asarray( [x for x in config.get('train', 'outsz').split(',') ], dtype=np.int64 )
    tpars['is_data_aug'] = config.getboolean('train', 'is_data_aug')
    tpars['is_rebalance']= config.getboolean('train', 'is_rebalance')
    tpars['is_malis']    = config.getboolean('train', 'is_malis')
    tpars['cost_fn_str'] = config.get('train', 'cost_fn')
    tpars['Num_iter_per_show'] = config.getint('train', 'Num_iter_per_show')
    tpars['Max_iter']    = config.getint('train', 'Max_iter')
    
    # forward parameters
    fpars['outsz']       = np.asarray( [x for x in config.get('forward', 'outsz').split(',') ], dtype=np.int64 )    
    fpars['out_prefix']  = config.get('forward','out_prefix').split(',\n')
    
    # cost function
    if tpars['cost_fn_str'] == "square_loss":
        tpars['cost_fn'] = cost_fn.square_loss
    elif tpars['cost_fn_str'] == "binomial_cross_entropy":
        tpars['cost_fn'] = cost_fn.binomial_cross_entropy
    elif tpars['cost_fn_str'] == "multinomial_cross_entropy":
        tpars['cost_fn'] = cost_fn.multinomial_cross_entropy 
    else:
        raise NameError('unknown type of cost function')
    
    #%% print parameters
    if tpars['is_rebalance']:
        print "rebalance the gradients"
    if tpars['is_malis']:
        print "using malis weight"
    return gpars, tpars, fpars

class CSamples:
    """class of sample, similar with Dataset module of pylearn2"""
    def __init__(self, vols, lbls):
        self.vols = vols
        self.lbls = lbls
    
    def read_tifs(ftrns, flbls=[], dp_type='volume'):
    """
    read a list of tif files of original volume and lable

    Parameters
    ----------
    ftrns:  list of file name of train volumes
    flbls:  list of file name of lable volumes

    Return
    ------
    vols:  list of training volumes
    lbls:  list of labeling volumes
    """
    # assert ( len(ftrns) == len(flbls) )
    vols = list()
    for ftrn in ftrns:
        vol = emirt.emio.imread(ftrn).astype('float32')
        # normalize the original volume
        vol = (vol - np.mean(vol)) / np.std(vol)
        vols.append( vol )
    if not flbls:
        return vols
    else:
        lbls = list()
        for flbl in flbls:
            lbl = emirt.emio.imread(flbl).astype('float32')
            # transform lable data to network output format
            if 'vol' in dp_type:
                lbl_net = lbl.reshape((1,)+lbl.shape).astype('float32')
            elif 'aff' in dp_type:
                lbl_net = np.zeros((3,)+lbl.shape, dtype='float32') 
                lbl_net[0,1:,:,:] = (lbl[1:,:,:] == lbl[:-1,:,:]) & (lbl[1:,:,:]>0)
                lbl_net[1,:,1:,:] = (lbl[:,1:,:] == lbl[:,:-1,:]) & (lbl[:,1:,:]>0)
                lbl_net[2,:,:,1:] = (lbl[:,:,1:] == lbl[:,:,:-1]) & (lbl[:,:,1:]>0)
            else:
                raise NameError("invalid data type name")
            lbls.append( lbl_net )
        return (vols, lbls)
    def get_sample(self, vols, insz, lbls, outsz):
        """
        get random sample from training and labeling volumes
    
        Parameters
        ----------
        vols :  list of training volumes.
        insz :  input size.
        lbls :  list of labeling volumes.
        outsz:  output size of network.
        type :  output data type: volume or affinity graph.
    
        Returns
        -------
        vol_ins  : input volume of network.
        vol_outs : label volume of network.
        """
        # pick random volume from volume list
        vid = np.random.randint( len(vols) )
        vol = vols[vid]
        lbl = lbls[vid]
        assert( vol.shape == lbl.shape[1:] )
        
        # configure size    
        half_in_sz  = insz.astype('uint32')  / 2
        half_out_sz = outsz.astype('uint32') / 2
       # margin consideration for even-sized input
       # margin_sz = (insz-1) / 2
        set_sz = vol.shape - insz + 1
        # get random location
        loc = np.zeros(3)
        
        self.vol_ins = np.empty(np.hstack((1,insz)), dtype='float32')
        self.lbl_outs= np.empty(np.hstack((3,outsz)), dtype='float32')
        loc[0] = np.random.randint(half_in_sz[0]+1, half_in_sz[0]+1 + set_sz[0])
        loc[1] = np.random.randint(half_in_sz[1]+1, half_in_sz[1]+1 + set_sz[1])
        loc[2] = np.random.randint(half_in_sz[2]+1, half_in_sz[2]+1 + set_sz[2])
        # extract volume
        self.vol_ins[0,:,:,:]  = vol[   loc[0]-half_in_sz[0]  : loc[0]-half_in_sz[0] + insz[0],\
                                        loc[1]-half_in_sz[1]  : loc[1]-half_in_sz[1] + insz[1],\
                                        loc[2]-half_in_sz[2]  : loc[2]-half_in_sz[2] + insz[2]]
        self.lbl_outs[:,:,:,:] = lbl[:, loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
                                        loc[1]-half_out_sz[1] : loc[1]-half_out_sz[1]+outsz[1],\
                                        loc[2]-half_out_sz[2] : loc[2]-half_out_sz[2]+outsz[2]]
        vol_ins = np.empty(np.hstack((1,insz)), dtype='float32')
        lbl_outs= np.empty(np.hstack((3,outsz)), dtype='float32')
        loc[0] = np.random.randint(half_in_sz[0], half_in_sz[0] + set_sz[0])
        loc[1] = np.random.randint(half_in_sz[1], half_in_sz[1] + set_sz[1])
        loc[2] = np.random.randint(half_in_sz[2], half_in_sz[2] + set_sz[2])
        # extract volume
        vol_ins[0,:,:,:]  = vol[    loc[0]-half_in_sz[0]  : loc[0]-half_in_sz[0] + insz[0],\
                                    loc[1]-half_in_sz[1]  : loc[1]-half_in_sz[1] + insz[1],\
                                    loc[2]-half_in_sz[2]  : loc[2]-half_in_sz[2] + insz[2]]
        lbl_outs[:,:,:,:] = lbl[ :, loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
                                    loc[1]-half_out_sz[1] : loc[1]-half_out_sz[1]+outsz[1],\
                                    loc[2]-half_out_sz[2] : loc[2]-half_out_sz[2]+outsz[2]]
        return (vol_ins, lbl_outs)


    def get_sparse_sample( vols, lbls, input_patch_shape, output_patch_shape ):
    
        #Select volume
        index = np.random.randint( len(vols) )
        vol = vols[index]
        lbl = lbls[index]
    
        # Shape of labels only containing valid locations for input patch
        valid_lbl_shape = lbl.shape - input_patch_shape + 1
        valid_lbl = volume_util.crop3d(lbl, valid_lbl_shape, round_up=True)
        
        input_patch_margin  = input_patch_shape  / 2 #rounds down
        output_patch_margin = output_patch_shape / 2 #ditto
    
        # Could be computationally expensive (may be worth modifying)
        nonzero_indices = np.nonzero(valid_lbl)
    
        next_offset_index = np.random_choice(nonzero_indices)
        next_index = next_offset_index + input_patch_margin
    
        input_vols = np.empty(np.hstack(1,input_patch_shape), dtype='float32')
        output_labels = np.empty(np.hstack(3,output_patch_shape), dtype='float32')
    
        input_vols[0,:,:,:] = vol[      next_index[0]-input_patch_margin[0]  :  
                                            next_index[0]-input_patch_margin[0]
                                            + input_patch_shape[0],
                                        next_index[1]-input_patch_margin[1]  :  
                                            next_index[1]-input_patch_margin[1]
                                            + input_patch_shape[1],
                                        next_index[2]-input_patch_margin[2]  :  
                                            next_index[2]-input_patch_margin[2]
                                            + input_patch_shape[2]]
        output_labels[:,:,:,:] = lbl[   next_index[0]-output_patch_margin[0] :
                                            next_index[0]-output_patch_margin
                                            + output_patch_shape[0],
                                        next_index[1]-output_patch_margin[1] :
                                            next_index[1]-output_patch_margin
                                            + output_patch_shape[1],
                                        next_index[2]-output_patch_margin[2] :
                                            next_index[2]-output_patch_margin
                                            + output_patch_shape[2]]
    
        return (input_vols, output_labels)



    def data_aug_transform(self, data, rft):
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

    def volume_aug(self, vols, lbls, dp_type = 'volume' ):
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
            vols[i,:,:,:] = data_aug_transform(vols[i,:,:,:], rft)
        for i in xrange(lbls.shape[0]):
            lbls[i,:,:,:] = self.data_aug_transform(lbls[i,:,:,:], rft)
        return (vols, lbls)

def inter_show(start, i, err, cls, it_list, err_list, cls_list, \
                terr_list, tcls_list, \
                eta, vol_ins, props, lbl_outs, grdts, tpars, \
                rb_weights=0, malis_weights=0, grdts_bm=0):
    # time
    elapsed = time.time() - start
    print "iteration %d,    err: %.3f,    cls: %.3f,   elapsed: %.1f s, learning rate: %.4f"\
            %(i, err, cls, elapsed, eta )
    # real time visualization
    plt.subplot(331),   plt.imshow(vol_ins[0,0,:,:],       interpolation='nearest', cmap='gray')
    plt.xlabel('input')
    plt.subplot(332),   plt.imshow(props[1,0,:,:],    interpolation='nearest', cmap='gray')
    plt.xlabel('inference')
    plt.subplot(333),   plt.imshow(lbl_outs[1,0,:,:], interpolation='nearest', cmap='gray')
    plt.xlabel('lable')
    plt.subplot(334),   plt.imshow(grdts[1,0,:,:],     interpolation='nearest', cmap='gray')
    plt.xlabel('gradient')


    plt.subplot(337), plt.plot(it_list, err_list, 'b', it_list, terr_list, 'r')
    plt.xlabel('iteration'), plt.ylabel('cost energy')
    plt.subplot(338), plt.plot(it_list, cls_list, 'b', it_list, tcls_list, 'r')
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
        plt.subplot(336),   plt.imshow( np.abs(grdts_bm[1,0,:,:] ),interpolation='nearest', cmap='gray')
        plt.xlabel('gradient befor malis')
    plt.pause(1)
    return start, err, cls

