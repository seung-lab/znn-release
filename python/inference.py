#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import numpy as np
import pyznn
import front_end
#%% parameters
gpars, tpars, fpars = front_end.parser( 'config.cfg' )
vol_orgs, lbl_orgs = front_end.read_tifs(tpars['ftrns'], tpars['flbls'])

#%% load network
net = front_end.load_network(gpars['fnet'], gpars['fnet_spec'], fpars['outsz'], gpars['num_threads'])

#%% run forward pass
