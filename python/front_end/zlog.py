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

def record_config_file(params=None, config_filename=None, net_save_filename=None,
    timestamp=None, train=True):
    '''
    Copies the config file used for the current run of ZNN under the same
    prefix as the network save prefix

    Format: {net_save_prefix}_{train/forward}_{timestamp}.cfg
    e.g. net_train_20151007_17:48:55.cfg
    '''
    import shutil

    #Need to specify either a params object, or all of the other optional args
    #"ALL" optional args excludes train
    utils.assert_arglist(params,
        [config_filename, net_save_filename]
        )

    #Args default to params values, override by options
    if params is not None:
        _config_filename = params['fconfig']
        if train:
            _net_save_filename = params['train_net']
        else:
            _net_save_filename = params['']

    #Option override
    if config_filename is not None:
        _config_filename = config_filename
    if net_save_filename is not None:
        _net_save_filename = net_save_filename

    #More error checking
    save_prefix = os.path.splitext( _net_save_filename )[0]

    config_file_exists = os.path.isfile( _config_filename )
    assert(config_file_exists)

    save_prefix_valid = len(save_prefix) > 0
    assert(save_prefix_valid)

    #Deriving destination filename information
    if timestamp is None:
        timestamp = utils.timestamp()
    mode = "train" if train else "forward"

    #Actually saving

    # get directory name from filename
    directory_name = os.path.dirname( save_prefix )
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    save_filename = "{}_{}_{}.cfg".format(save_prefix, mode, timestamp)
    shutil.copy( _config_filename, save_filename)

def make_logfile_name(params=None, net_save_filename=None, timestamp = None, train=True):
    '''
    Returns the name of the logfile for the current training/forward pass run
    '''

    #Need to specify either a params object, or the net save prefix
    utils.assert_arglist(params,
        [net_save_filename])

    if params is not None:
        if train:
            _net_save_filename = params['train_net']
        else:
            _net_save_filename = params['output_prefix']

    if net_save_filename is not None:
        _net_save_filename = net_save_filename

    save_prefix = os.path.splitext( _net_save_filename )[0]

    save_prefix_valid = len(save_prefix) > 0
    assert(save_prefix_valid)

    if timestamp is None:
        timestamp = utils.timestamp()
    mode = "train" if train else "forward"

    directory_name = os.path.dirname( save_prefix )
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    save_filename = "{}_{}_{}.log".format(save_prefix, mode, timestamp)

    return save_filename
