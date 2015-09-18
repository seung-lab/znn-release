#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
from matplotlib.pylab import plt

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
    plt.show()
    
if __name__ == '__main__':
    import sys
    if len(sys.argv)>1:
        fname = sys.argv[1]
    else:
        fname = 'ARCHIVE/net_statistics.h5'
        
    show_net_statistics( fname )