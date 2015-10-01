#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
from matplotlib.pylab import plt

#%% smooth function
def smooth(x, y, w):
    # averaging the curve
    x = np.asarray(x)
    y = np.asarray(y)
    w = int(w)
    assert(w>0)
    assert(x.ndim==1)
    assert(y.ndim==1)
    lw = (w-1)/2
    rw = w/2

    x2 = list()
    y2 = list()
    for i in xrange(lw, x.size-rw, w):
        x2.append( x[i] )
        y2.append( np.mean( y[i-lw:i+rw+1] ) )
    return x2, y2

def show_net_statistics( fname, w ):
    """
    illustrate the learning curve

    Parameters
    ----------
    fname : the statistics h5 file name
    w : int, window size for smoothing the curve
    """
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
    plt.plot(tn_it, tn_err, 'b.', alpha=0.2)
    plt.plot(tt_it, tt_err, 'r.', alpha=0.2)
    # plot smoothed line
    xne,yne = smooth(tn_it, tn_err, w)
    xte,yte = smooth(tt_it, tt_err, w)
    plt.plot(xne, yne, 'b')
    plt.plot(xte, yte, 'r')
    plt.xlabel('iteration'), plt.ylabel('cost energy')

    plt.subplot(122)
    plt.plot(tn_it, tn_cls, 'b.', alpha=0.2)
    plt.plot(tt_it, tt_cls, 'r.', alpha=0.2)
    # plot smoothed line
    xnc, ync = smooth(tn_it, tn_cls, w)
    xtc, ytc = smooth(tt_it, tt_cls, w)
    plt.plot(xnc, ync, 'b', label='train')
    plt.plot(xtc, ytc, 'r', label='test')
    plt.xlabel('iteration'), plt.ylabel( 'classification error' )
    plt.legend()
    plt.show()

if __name__ == '__main__':
    import sys
    # default window size
    w = 3

    if len(sys.argv)==2:
        fname = sys.argv[1]
    elif len(sys.argv)==3:
        fname = sys.argv[1]
        w = int( sys.argv[2] )
        print "window size: ", w
    else:
        raise NameError("no input statistics h5 file!")

    show_net_statistics( fname,  w)