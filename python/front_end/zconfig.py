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
import utils
from emirt import volume_util
import zaws

def parser(conf_fname):
    # parse config file to get parameters
    pars = parse_cfg(conf_fname)
    # correct some parameter setting
    pars = autoset_pars(pars)
    # checking and automatically correcting parameters
    check_pars(pars)

    # dataset spec
    dspec = parse_data_spec(pars['fdata_spec'])
    dspec = autoset_dspec(pars, dspec)
    return dspec, pars

def parse_cfg( conf_fname ):
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
    if config.has_option('parameters', 'train_net_prefix'):
        pars['train_net_prefix'] = config.get('parameters', 'train_net_prefix')
    elif config.has_option('parameters', 'train_net'):
        pars['train_net_prefix'] = config.get('parameters', 'train_net')
        # remove the ".h5"
        import string
        pars['train_net_prefix'] = string.replace(pars['train_net_prefix'], ".h5", "")
    elif config.has_option('parameters', 'train_save_net'):
        pars['train_net_prefix'] = config.get('parameters', 'train_save_net')
        # remove the ".h5"
        import string
        pars['train_net_prefix'] = string.replace(pars['train_net_prefix'], ".h5", "")

    #Whether to write .log and .cfg files
    if config.has_option('parameters', 'logging'):
        pars['logging'] = config.getboolean('parameters', 'logging')

    #TRAINING OPTIONS
    #Samples to use for training
    if config.has_option('parameters', 'train_range'):
        pars['train_range'] = utils.parseIntSet( config.get('parameters',   'train_range') )

    #Samples to use for cross-validation
    if config.has_option('parameters', 'test_range'):
        pars['test_range']  = utils.parseIntSet( config.get('parameters',   'test_range') )
    #Learning Rate
    if config.has_option('parameters', 'eta'):
        pars['eta']         = config.getfloat('parameters', 'eta')
    else:
        pars['eta'] = 0.01
    #Learning Rate Annealing Factor
    if config.has_option('parameters', 'anneal_factor'):
        pars['anneal_factor'] = config.getfloat('parameters', 'anneal_factor')
    else:
        pars['anneal_factor'] = 1
    #Momentum Constant
    if config.has_option('parameters', 'momentum'):
        pars['momentum']    = config.getfloat('parameters', 'momentum')
    else:
        pars['momentum'] = 0
    #Weight Decay
    if config.has_option('parameters', 'weight_decay'):
        pars['weight_decay']= config.getfloat('parameters', 'weight_decay')
    else:
        pars['weight_decay'] = 0
    #Training Output Patch Shape
    if config.has_option('parameters', 'train_outsz'):
        pars['train_outsz'] = np.asarray( [x for x in config.get('parameters', \
                                    'train_outsz').split(',') ], dtype=np.int64 )
    else:
        pars['train_outsz'] = np.array([1,100,100])
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
    if config.has_option('parameters', 'is_data_aug'):
        pars['is_data_aug'] = config.getboolean('parameters', 'is_data_aug')
    else:
        pars['is_data_aug'] = False
    # simulate jitter in real image stack
    if config.has_option('parameters', 'jitter_size'):
        pars['jitter_size'] = config.getint('parameters', 'jitter_size')
    else:
        pars['jitter_size'] = 0
    #Whether to use boundary mirroring
    if config.has_option('parameters', 'is_bd_mirror'):
        pars['is_bd_mirror'] = config.getboolean('parameters', 'is_bd_mirror')
    else:
        pars['is_bd_mirror'] = False
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
    if config.has_option('parameters', 'cost_fn'):
        pars['cost_fn_str'] = config.get('parameters', 'cost_fn')
    else:
        pars['cost_fn_str'] = 'auto'

    #DISPLAY OPTIONS
    #How often to show progress to the screen
    if config.has_option('parameters', 'Num_iter_per_show'):
        pars['Num_iter_per_show'] = config.getint('parameters', 'Num_iter_per_show')
    else:
        pars['Num_iter_per_show'] = 100
    #How often to check cross-validation error
    if config.has_option('parameters', 'Num_iter_per_test'):
        pars['Num_iter_per_test'] = config.getint('parameters', 'Num_iter_per_test')
    else:
        pars['Num_iter_per_test'] = 500
    #How many output patches should derive cross-validation error
    if config.has_option('parameters', 'test_num'):
        pars['test_num'] = config.getint( 'parameters', 'test_num' )
    else:
        pars['test_num'] = 10
    #How often to save the network
    if config.has_option('parameters', 'Num_iter_per_save'):
        pars['Num_iter_per_save'] = config.getint('parameters', 'Num_iter_per_save')
    else:
        pars['Num_iter_per_save'] = 1000
    #How often to change the learning rate
    if config.has_option('parameters','Num_iter_per_annealing'):
        pars['Num_iter_per_annealing'] = config.getint('parameters', 'Num_iter_per_annealing')
    else:
        pars['Num_iter_per_annealing'] = 100
    #Maximum training updates
    if config.has_option('parameters', 'Max_iter'):
        pars['Max_iter'] = config.getint('parameters', 'Max_iter')
    else:
        pars['Max_iter'] = 400000

    #FULL FORWARD PASS PARAMETERS
    #Which samples to use
    pars['forward_range'] = utils.parseIntSet( config.get('parameters', 'forward_range') )
    #Which network file to load
    pars['forward_net']   = config.get('parameters', 'forward_net')
    #Output Patch Size
    pars['forward_outsz'] = np.asarray( [x for x in config.get('parameters', 'forward_outsz')\
                                        .split(',') ], dtype=np.int64 )
    #Prefix of the output files
    pars['output_prefix'] = config.get('parameters', 'output_prefix')

    return pars

def autoset_pars(pars):
    #PROCESSING COST FUNCTION STRING
    if 'auto' in pars['cost_fn_str']:
        # automatic choosing of cost function
        if 'boundary' in pars['out_type']:
            pars['cost_fn_str'] = 'softmax_loss'
            pars['cost_fn'] = cost_fn.softmax_loss
        elif 'affin' in pars['out_type']:
            pars['cost_fn_str'] = 'binomial_cross_entropy'
            pars['cost_fn'] = cost_fn.binomial_cross_entropy
        elif 'semantic' in pars['out_type']:
            pars['cost_fn_str'] = 'softmax_loss'
            pars['cost_fn'] = cost_fn.softmax_loss
        else:
            raise NameError("no matching cost function for out_type!")
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

    # aws s3 filehandling
    pars['fnet_spec']  = zaws.s3download( pars['fnet_spec'] )
    pars['fdata_spec'] = zaws.s3download( pars['fdata_spec'] )
    # local file name
    if "s3://" in pars['train_net_prefix']:
        # copy the path as a backup
        pars['s3_train_net_prefix'] = pars['train_net_prefix']
        bn = os.path.basename( pars['train_net_prefix'] )
        # replace with local path
        pars['train_net_prefix'] = "/tmp/{}".format(bn)
    return pars

def check_pars(pars):
    """
    check and correct the configuration and parameters

    Parameters
    ----------
    config : python parser reading of config file.
    pars : the parameters.
    """

    # check the single parameters
    assert(pars['num_threads']>=0)
    assert('float32'==pars['dtype'] or 'float64'==pars['dtype'])
    assert('boundary' in pars['out_type'] or 'affin' in pars['out_type']) or 'semantic' in pars['out_type']
    assert( np.size(pars['train_outsz'])==3 )
    assert(pars['anneal_factor']>=0 and pars['anneal_factor']<=1)
    assert(pars['momentum']>=0      and pars['momentum']<=1)
    assert(pars['weight_decay']>=0  and pars['weight_decay']<=1)

    # normally, we shoud not use two rebalance technique together
    assert (pars['rebalance_mode'] is None) or ('global' in pars['rebalance_mode']) or ('patch' in pars['rebalance_mode'] or ('none' in pars['rebalance_mode']))

    assert(pars['Num_iter_per_show']>0)
    assert(pars['Num_iter_per_test']>0)
    assert(pars['test_num']>0)
    assert(pars['Num_iter_per_save']>0)
    assert(pars['Max_iter']>0)
    assert(pars['Max_iter']>pars['Num_iter_per_save'])

    # check malis normalization type
    if pars['is_malis']:
        assert 'none' in pars['malis_norm_type'] \
            or 'frac' in pars['malis_norm_type'] \
            or 'num'  in pars['malis_norm_type'] \
            or 'pair' in pars['malis_norm_type'] \
            or 'constrain' in pars['malis_norm_type']


def parse_data_spec(fdata_spec):
    # initialize data spec dict
    dspec = dict()
    # read specification file
    config = ConfigParser.ConfigParser()
    config.read( fdata_spec )

    for sec in config.sections():
        dspec[sec] = dict()
        for opt in config.options(sec):
            dspec[sec][opt] = config.get(sec, opt)
            # backward compatable for indexed image or label
            if 'sample' in sec  and dspec[sec][opt].isdigit():
                if 'in' in opt:
                    dspec[sec][opt] = 'image' + dspec[sec][opt]
                elif 'out' in opt:
                    dspec[sec][opt] = 'label' + dspec[sec][opt]
                else:
                    raise NameError('unsupported input and output name style in dataset specification file!')
    return dspec

# auto correct the image and labels
def autoset_dspec(pars, dspec):
    for sec in dspec.keys():
        if 'label' in sec:
            pp_types = dspec[sec]['pp_types']
            if 'boundary' in pars['out_type']:
                pp_types = pp_types.replace("auto", "binary_class")
            elif 'affin' in pars['out_type']:
                pp_types = pp_types.replace("auto", "affinity")
            dspec[sec]['pp_types'] = pp_types
    return dspec

# parse args
def parse_args(args):
    # s3 to local
    args['config'] = zaws.s3download( args['config'] )
    args['seed'] = zaws.s3download( args['seed'] )
    #%% parameters
    if not os.path.exists( args['config'] ):
        raise NameError("config file not exist!")
    else:
        print "reading config parameters..."
        dspec, pars = parser( args["config"] )

    # overwrite the config file parameters from command line
    if args["is_check"]:
        if "yes" == args["is_check"]:
            pars["is_check"] = True
        elif "no" == args["is_check"]:
            pars["is_check"] = False
        else:
            raise NameError("invalid checking option in command line")
    # data type
    if args["dtype"]:
        if "single" in args["dtype"] or "float32" in args["dtype"]:
            pars["dtype"] = "float32"
        elif "double" in args["dtype"] or "float64" in args["dtype"]:
            pars["dtype"] = "float64"
        else:
            raise NameError("invalid data type defined in command line.")

    # random seed
    if pars['is_debug'] or pars['is_check']:
        # use fixed index
        np.random.seed(1)

    if pars.has_key('logging') and pars['logging']:
        print "recording configuration file..."
        zlog.record_config_file( pars )
        logfile = zlog.make_logfile_name( pars )
    else:
        logfile = None

    # check the seed file
    if args['seed']:
        if not os.path.exists(args['seed']):
            import warnings
            warnings.warn("seed file not found! use train_net_prefix of configuration instead.")
        else:
            pars['seed'] = args['seed']
    else:
        pars['seed'] = None
    return dspec, pars, logfile
