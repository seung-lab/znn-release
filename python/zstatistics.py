#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
from matplotlib.pylab import plt
from os import path

class CLearnCurve:
    def __init__(self, fname=None):
        # initialize the key-value store
        self.train = dict()
        self.test  = dict()
        if fname is not None:
            self._read_curve( fname )
        return

    def _read_curve(self, fname):
        # initialize by loading from a h5 file
        # get the iteration number
        iter_num = self._get_iter_num(fname)

        assert( path.exists(fname) )
        # read data
        import h5py
        # read into memory
        f = h5py.File(fname, 'r', driver='core')

        if "/processing/znn/train/statistics/" in f:
            self.stdpre = "/processing/znn/train/statistics/"
            print "stdpre: ", self.stdpre
        else:
            self.stdpre = "/"

        print "stdpre: ", self.stdpre
        ft = f[self.stdpre + 'test']
        for key in ft:
            self.test[key] = ft[key].value

        ft = f[self.stdpre + 'train/']
        for  key in ft:
            self.train[key] = ft[key].value

        f.close()

        # crop the values
        if iter_num is not None:
            self._crop_iters(iter_num)
        return

    def _crop_iters(self, iter_num):

        # test iterations
        gen = (i for i,v in enumerate(self.test['it']) if v>iter_num)
        try:
            ind = next(gen)
            for key,value in self.test.items():
                self.test[key] = self.test[key][:ind]
        except StopIteration:
            pass

        # train iterations
        gen = (i for i,v in enumerate(self.train['it']) if v>iter_num)
        try:
            ind = next(gen)
            for key, value in self.train.items():
                self.train[key] = self.train[key][:ind]
        except StopIteration:
            pass

        return

    def _get_iter_num(self, fname ):
        if '.h5' not in fname:
            return None
        root, ext = path.splitext(fname)
        str_num = root.split('_')[-1]
        if 'current' in str_num :
            # the last network
            return None
        else:
            return int(str_num)

    def append_test(self, it, err, cls, re):
        # add a test result
        self.test['it'].append(it)
        self.test['err'].append(err)
        self.test['cls'].append(cls)
        self.test['re'].append(re)

    def append_train(self, it, err, cls, re):
        # add a training result
        self.train['it'].append(it)
        self.train['err'].append(err)
        self.train['cls'].append(cls)
        self.train['re'].append(re)

    def get_last_it(self):
        # return the last iteration number
        if len(self.test['it'])>0 and len(self.train['it'])>0:
            last_it = max( self.test['it'][-1], self.train['it'][-1] )
            print "inherit last iteration: ", last_it
            return last_it
        else:
            return 0

    #%% smooth function
    def _smooth(self, x, y, w):
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

    def _find_max_update(self, it, vec):
        """
        find the maximum iteration without nan
        """
        i = len(vec)-1
        # traverse from end to start
        for v in vec[::-1]:
            if v is not np.nan:
                return it[i]
            i -= 1
        return 0

    def print_max_update(self):
        print "max iter: ", self._find_max_update( self.train['it'], self.train['cls'] )

    def plot(self, w=3):
        """
        illustrate the learning curve

        Parameters
        ----------
        w : int, window size for smoothing the curve
        """
        # number of subplots
        nsp = len(self.train)-1
        print "number of subplots: {}".format(nsp)

        # print the maximum iteration
        self.print_max_update()

        # using K as iteration unit
        tn_it = self.train['it']
        for i in xrange(len(tn_it)):
            tn_it[i] = tn_it[i] / float(1000)
        tt_it = self.test['it']
        for i in xrange(len(tt_it)):
            tt_it[i] = tt_it[i] / float(1000)

        # plot data
        idx = 0
        for key in self.train.keys():
            if key == 'it':
                continue
            idx += 1
            plt.subplot(1,nsp, idx)
            plt.plot(self.train['it'], self.train[key], 'b.', alpha=0.2)
            plt.plot(self.test['it'],  self.test[key],  'r.', alpha=0.2)
            # plot smoothed line
            xne,yne = self._smooth( self.train['it'], self.train[key], w )
            xte,yte = self._smooth( self.test['it'],  self.test[key],  w )
            plt.plot(xne, yne, 'b')
            plt.plot(xte, yte, 'r')
            plt.xlabel('iteration (K)'), plt.ylabel(key)

        plt.legend()
        plt.show()
        return

    def save(self, pars, fname=None, elapsed=0, suffix=None):
        if pars['is_stdio']:
            self.stdpre = "/processing/znn/train/statistics/"
            print "stdpre: ", self.stdpre
        else:
            self.stdpre = "/"

        if not pars['is_stdio']:
             # change filename
            root = pars['train_net_prefix']
            import os
            import shutil

            #storing in case of a suffix,
            # so 'current' file below isn't duplicated
            orig_root = root
            if suffix is not None:
                root = "{}_{}".format(pars['train_net_prefix'], suffix)

            if len(self.train['it']) > 0:
                fname = root + '_statistics_{}.h5'.format( self.train['it'][-1] )
            else:
                fname = root + '_statistics_0.h5'
            if os.path.exists(fname):
                os.remove( fname )

        # save variables
        import h5py
        f = h5py.File( fname, 'a' )
        for key, value in self.train.items():
            f.create_dataset(self.stdpre + '/train/' + key,  data=value )
        for key, value in self.test.items():
            f.create_dataset(self.stdpre + '/test/' + key,  data=value)

        f.create_dataset(self.stdpre + '/elapsed',   data=elapsed)
        f.close()

        if not pars['is_stdio']:
            # move to new name
            fname2 = root + '_statistics_current.h5'
            if os.path.exists( fname2 ):
                os.remove( fname2 )
            shutil.copyfile(fname, fname2)

if __name__ == '__main__':
    """
    show learning curve

    usage
    ----
    python statistics.py path/of/statistics.h5 5
    5 is an example of smoothing window size
    """
    import sys
    # default window size
    w = 3
    assert len(sys.argv) > 1
    fname = sys.argv[1]

    lc = CLearnCurve( fname )

    if len(sys.argv)==3:
        w = int( sys.argv[2] )
        print "window size: ", w
    lc.plot( w )
