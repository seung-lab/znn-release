#!/usr/bin/env python
__doc__ = """

Front-End Interface for ZNNv4

Jingpeng Wu <jingpeng.wu@gmail.com>,
Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""

import ConfigParser

import numpy as np
import matplotlib.pylab as plt

import cost_fn
from Samples import *

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

def show_net_statistics( fname ):
    # read data
    import h5py
    f = h5py.File(fname)
    tt_it  = np.asarray( f['/test/it']  )
    tt_err = np.asarray( f['/test/err'] )
    tt_cls = np.asarray( f['/test/cls'] )
    
    tn_it  = np.asarray( f['/train/it'] )
    tn_err = np.asarray( f['/train/err'])
    tn_cls = np.asarray( f['/train/cls'])
    f.close()
    
    # plot data
    plt.subplot(121)
    plt.plot(tn_it, tn_err, 'b', label='train')
    plt.plot(tt_it, tt_err, 'r', label='test')
    plt.xlabel('iteration'), plt.ylabel('cost energy')
    plt.subplot(122)
    plt.plot(tn_it, tn_cls, 'b', label='train')
    plt.plot(tt_it, tt_cls, 'r', label='test')
    plt.xlabel('iteration'), plt.ylabel( 'classification error' )
    plt.legend()