#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import numpy as np
import pyznn
import front_end
#%% parameters
gpars, tpars, fpars = front_end.parser( 'config.cfg' )
# read image stacks
vol_orgs = front_end.read_tifs(fpars['ffwds'])

#%% load network
net = front_end.load_network(gpars['fnet'], gpars['fnet_spec'], fpars['outsz'], gpars['num_threads'])

# get input size
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1

#%% run forward pass
half_in_sz  = insz.astype('uint32')  / 2
half_out_sz = outsz.astype('uint32') / 2
# margin consideration for even-sized input
margin_sz = half_in_sz - (insz%2)

# initialize the output volume
vol_outs = list()

for vol in vol_orgs:
    set_sz = vol.shape - margin_sz - half_in_sz
    # output volume
    vol_out = np.empty(set_sz)
    # sliding the volume
    for x1 in xrange