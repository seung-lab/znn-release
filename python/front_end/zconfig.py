#!/usr/bin/env python
__doc__ = """

Front-End Interface for ZNNv4

Jingpeng Wu <jingpeng.wu@gmail.com>,
Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""

import ConfigParser
import numpy as np
import os
import cost_fn
import zsample, zsample_thr
import utils

from emirt import volume_util

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
    if nputstr is None:
        return None

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
    #Config filename
    pars['fconfig'] = conf_fname
    #Network specification filename
    pars['fnet_spec']   = config.get('parameters', 'fnet_spec')
    if config.has_option('parameters', 'fdata_spec'):
        pars['fdata_spec'] = config.get('parameters', 'fdata_spec')

    #Number of threads to use
    pars['num_threads'] = int( config.get('parameters', 'num_threads') )
    if pars['num_threads'] <= 0:
        # use maximum number of cpus
        import multiprocessing
        pars['num_threads'] = multiprocessing.cpu_count()
    # data type
    pars['dtype']       = config.get('parameters', 'dtype')
    #Output layer data type (e.g. 'boundary','affinity')
    pars['out_type']   = config.get('parameters', 'out_type')


    #IO OPTIONS
    #Filename under which we save the network
    if config.has_option('parameters', 'train_net'):
        pars['train_net'] = config.get('parameters', 'train_net')
    elif config.has_option('parameters', 'train_save_net'):
        pars['train_net'] = config.get('parameters', 'train_save_net')
    #Whether to write .log and .cfg files
    if config.has_option('parameters', 'logging'):
        pars['logging'] = config.getboolean('parameters', 'logging')


    #TRAINING OPTIONS
    #Samples to use for training
    pars['train_range'] = parseIntSet( config.get('parameters',   'train_range') )
    #Samples to use for cross-validation
    pars['test_range']  = parseIntSet( config.get('parameters',   'test_range') )
    #Learning Rate
    pars['eta']         = config.getfloat('parameters', 'eta')
    #Learning Rate Annealing Factor
    if config.has_option('parameters', 'anneal_factor'):
        pars['anneal_factor'] = config.getfloat('parameters', 'anneal_factor')
    else:
        pars['anneal_factor'] = 1
    #Momentum Constant
    pars['momentum']    = config.getfloat('parameters', 'momentum')
    #Weight Decay
    pars['weight_decay']= config.getfloat('parameters', 'weight_decay')
    #Training Output Patch Shape
    pars['train_outsz'] = np.asarray( [x for x in config.get('parameters', \
                                    'train_outsz').split(',') ], dtype=np.int64 )
    #Whether to optimize the convolution computation by layer
    # (FFT vs Direct Convolution)
    if config.has_option("parameters", "is_train_optimize"):
        if config.getboolean('parameters', 'is_train_optimize'):
            pars['train_conv_mode'] = "optimize"
        else:
            pars['train_conv_mode'] = "direct"
    if config.has_option("parameters", "is_forward_optimize"):
        if config.getboolean('parameters', 'is_forward_optimize'):
            pars['forward_conv_mode'] = "optimize"
        else:
            pars['forward_conv_mode'] = 'direct'
    if config.has_option('parameters', 'force_fft'):
        if config.getboolean('parameters', 'force_fft'):
            pars['train_conv_mode'] = 'fft'
            pars['forward_conv_mode'] = 'fft'
    if config.has_option('parameters', 'train_conv_mode'):
        pars['train_conv_mode'] = config.get('parameters', 'train_conv_mode')
    if config.has_option('parameters', 'forward_conv_mode'):
        pars['forward_conv_mode'] = config.get('parameters', 'forward_conv_mode')

    #Whether to use data augmentation
    pars['is_data_aug'] = config.getboolean('parameters', 'is_data_aug')
    #Whether to use boundary mirroring
    pars['is_bd_mirror']= config.getboolean('parameters', 'is_bd_mirror')
    
    #Whether to use balanced sampling (50%pos/neg)
    if config.has_option('parameters', 'balanced_sampling'):
        pars['balanced_sampling'] = config.getboolean('parameters','balanced_sampling')
    else:
        pars['balanced_sampling'] = False

    #Whether to use rebalanced training
    if config.has_option('parameters', 'is_rebalance'):
        if config.getboolean('parameters', 'is_rebalance'):
            pars['rebalance_mode'] = 'global'
        else:
            pars['rebalance_mode'] = None
    # whether to use rebalance of output patch
    if config.has_option('parameters', 'is_patch_rebalance'):
        if config.getboolean('parameters', 'is_patch_rebalance'):
            pars['rebalance_mode'] = 'patch'
        else:
            pars['rebalance_mode'] = None
    if config.has_option('parameters', 'rebalance_mode'):
        pars['rebalance_mode'] = config.get('parameters', 'rebalance_mode')
    else:
        pars['rebalance_mode'] = None

    #Whether to use malis cost
    if config.has_option('parameters', 'is_malis'):
        pars['is_malis'] = config.getboolean('parameters', 'is_malis')
    else:
        pars['is_malis'] = False

    # malis normalization type
    if config.has_option('parameters', 'malis_norm_type'):
        pars['malis_norm_type'] = config.get( 'parameters', 'malis_norm_type' )
    else:
        pars['malis_norm_type'] = 'none'
    # constrained malis
    if config.has_option('parameters', 'is_constrained_malis'):
        pars['is_constrained_malis'] = config.getboolean('parameters', 'is_constrained_malis')
    else:
        pars['is_constrained_malis'] = False
    #Data Provider
    if config.has_option('parameters','data_provider'):
        pars['data_provider'] = config.get( 'parameters', 'data_provider')
    else:
        pars['data_provider'] = 'default' #processed below

    #Whether to display progress plots
    if config.has_option('parameters', "is_visual"):
        pars['is_visual']   = config.getboolean('parameters', 'is_visual')
    else:
        pars['is_visual'] = False

    # standard IO
    if config.has_option('parameters', 'is_stdio'):
        pars['is_stdio'] = config.getboolean('parameters', 'is_stdio')
    else:
        pars['is_stdio'] = False
    # debug mode
    if config.has_option('parameters', 'is_debug'):
        pars['is_debug'] = config.getboolean('parameters', 'is_debug')
    else:
        pars['is_debug'] = False
    # automatically check the gradient and patch matching
    if config.has_option("parameters", "is_check"):
        pars["is_check"] = config.getboolean("parameters", "is_check")
    else:
        pars["is_check"] = False

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
    #How often to change the learning rate
    if config.has_option('parameters','Num_iter_per_annealing'):
        pars['Num_iter_per_annealing'] = config.getint('parameters', 'Num_iter_per_annealing')
    else:
        pars['Num_iter_per_annealing'] = 100
    #Maximum training updates
    pars['Max_iter']    = config.getint('parameters', 'Max_iter')
    #Numer of Iterations between dataset swaps
    if config.has_option('parameters','Num_iter_per_dset_swap'):
        pars['Num_iter_per_dset_swap'] = config.getint('parameters', 'Num_iter_per_dset_swap')

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


    if 'fdata_spec' in pars.keys():
        assert( os.path.exists( pars['fdata_spec'] ) )
        config.read( pars['fdata_spec'] )
    # checking and automatically correcting parameters
    config, pars = check_config(config, pars)

    return config, pars

def check_config(config, pars):
    """
    check and correct the configuration and parameters

    Parameters
    ----------
    config : python parser reading of config file.
    pars : the parameters.
    """
    #PROCESSING COST FUNCTION STRING
    if 'auto' in pars['cost_fn_str']:
        # automatic choosing of cost function
        if 'boundary' in pars['out_type']:
            pars['cost_fn_str'] = 'softmax_loss'
            pars['cost_fn'] = cost_fn.softmax_loss
        elif 'affin' in pars['out_type']:
            pars['cost_fn_str'] = 'binomial_cross_entropy'
            pars['cost_fn'] = cost_fn.binomial_cross_entropy
    elif "square-square" in pars['cost_fn_str']:
        pars['cost_fn'] = cost_fn.square_square_loss
    elif "square" in pars['cost_fn_str']:
        pars['cost_fn'] = cost_fn.square_loss
    elif  "binomial" in pars['cost_fn_str']:
        pars['cost_fn'] = cost_fn.binomial_cross_entropy
    elif "softmax" in pars['cost_fn_str']:
        pars['cost_fn'] = cost_fn.softmax_loss
    else:
        raise NameError('unknown type of cost function')

    #PROCESSING DATA PROVIDER STRING
    if ('S3' in pars['data_provider'] or
        's3' in pars['data_provider'] ):
        pars['data_provider'] = zsample_thr.CThreadedSamples_S3
    elif 'threaded' in pars['data_provider']:
        pars['data_provider'] = zsample_thr.CThreadedSamples
    else:
        pars['data_provider'] = zsample.CSamples

    # check the single parameters
    assert(pars['num_threads']>=0)
    assert('float32'==pars['dtype'] or 'float64'==pars['dtype'])
    assert('boundary' in pars['out_type'] or 'affin' in pars['out_type'])
    assert( np.size(pars['train_outsz'])==3 )
    assert(pars['anneal_factor']>=0 and pars['anneal_factor']<=1)
    assert(pars['momentum']>=0      and pars['momentum']<=1)
    assert(pars['weight_decay']>=0  and pars['weight_decay']<=1)

    # normally, we shoud not use two rebalance technique together
    assert (pars['rebalance_mode'] is None) or ('global' in pars['rebalance_mode']) or ('patch' in pars['rebalance_mode'])

    assert(pars['Num_iter_per_show']>0)
    assert(pars['Num_iter_per_test']>0)
    assert(pars['test_num']>0)
    assert(pars['Num_iter_per_save']>0)
    assert(pars['Max_iter']>0)
    assert(pars['Max_iter']>pars['Num_iter_per_save'])

    # check and correct the image and labels
    for sec in config.sections():
        if 'label' in sec:
            pp_types = config.get(sec, 'pp_types')
            if 'boundary' in pars['out_type']:
                pp_types = pp_types.replace("auto", "binary_class")
            elif 'affin' in pars['out_type']:
                pp_types = pp_types.replace("auto", "affinity")
            config.set(sec, 'pp_types', value=pp_types)

    #Remove fnames and fmasks options with ''
    for sec in config.sections():
        if config.has_option(sec, 'fnames') and len( config.get(sec, 'fnames') ) == 0:
            config.remove_option(sec, 'fnames')
        if config.has_option(sec, 'fmasks') and len( config.get(sec, 'fmasks') ) == 0:
            config.remove_option(sec, 'fmasks')

    # check malis normalization type
    if pars['is_malis']:
        assert 'none' in pars['malis_norm_type'] \
            or 'frac' in pars['malis_norm_type'] \
            or 'num'  in pars['malis_norm_type'] \
            or 'pair' in pars['malis_norm_type'] \
            or 'constrain' in pars['malis_norm_type']
    return config, pars
