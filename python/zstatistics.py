#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
from matplotlib.pylab import plt
from os import path

class CLearnCurve:
    def __init__(self, fname=None):
        if fname is None:
            # initialize with empty list
            self.tt_it  = list()
            self.tt_err = list()
            self.tt_cls = list()
            self.tt_re  = list()
            self.tt_mc  = list() # malis weighted cls
            self.tt_me  = list() # malis weighted cost energy

            self.tn_it  = list()
            self.tn_err = list()
            self.tn_cls = list()
            self.tn_re  = list()
            self.tn_mc  = list() # malis weighted cls
            self.tn_me  = list() # malis weighted cost energy
        else:
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
        self.tt_it  = list( f[self.stdpre + 'test/it'].value )
        self.tt_err = list( f[self.stdpre + 'test/err'].value )
        self.tt_cls = list( f[self.stdpre + 'test/cls'].value )
        if self.stdpre+'/test/re' in f:
            self.tt_re = list( f[self.stdpre + 'test/re'].value )
        else:
            self.tt_re = list()
        if self.stdpre + '/test/mc' in f:
            self.tt_mc = list( f[self.stdpre + 'test/mc'].value )
        else:
            self.tt_mc = list()
        if self.stdpre + '/test/me' in f:
            self.tt_me = list( f[self.stdpre + '/test/me'].value )
        else:
            self.tt_me = list()

        self.tn_it  = list( f[self.stdpre + '/train/it'].value )
        self.tn_err = list( f[self.stdpre + '/train/err'].value )
        self.tn_cls = list( f[self.stdpre + '/train/cls'].value )
        if self.stdpre + '/train/re' in f:
            self.tn_re = list( f[self.stdpre + '/train/re'].value )
        else:
            self.tn_re = list()

        if self.stdpre + '/train/mc' in f:
            self.tn_mc = list( f[self.stdpre + '/train/mc'].value )
        else:
            self.tn_mc = list()

        if self.stdpre + '/train/me' in f:
            self.tn_me = list( f[self.stdpre + '/train/me'].value )
        else:
            self.tn_me = list()

        f.close()

        # crop the values
        if iter_num is not None:
            self._crop_iters(iter_num)
        return

    def _crop_iters(self, iter_num):

        # test iterations
        gen = (i for i,v in enumerate(self.tt_it) if v>iter_num)
        try:
            ind = next(gen)
            self.tt_it  = self.tt_it[:ind]
            self.tt_err = self.tt_err[:ind]
            self.tt_cls = self.tt_cls[:ind]
        except StopIteration:
            pass

        # train iterations
        gen = (i for i,v in enumerate(self.tn_it) if v>iter_num)
        try:
            ind = next(gen)
            self.tn_it  = self.tn_it[:ind]
            self.tn_err = self.tn_err[:ind]
            self.tn_cls = self.tn_cls[:ind]
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
        self.tt_it.append(it)
        self.tt_err.append(err)
        self.tt_cls.append(cls)
        self._append_test_rand_error( re )

    def append_train(self, it, err, cls, re):
        # add a training result
        self.tn_it.append(it)
        self.tn_err.append(err)
        self.tn_cls.append(cls)
        self._append_train_rand_error( re )

    def _append_train_rand_error( self, re ):
        while len( self.tn_re ) < len(self.tn_it)-1:
            # fill with nan
            self.tn_re.append( np.nan )
        self.tn_re.append( re )
        assert len(self.tn_it) == len(self.tn_re)

    def append_train_malis_cls( self, mc ):
        while len( self.tn_mc ) < len(self.tn_it)-1:
            # fill with nan
            self.tn_mc.append( np.nan )
        self.tn_mc.append( mc )
        assert len(self.tn_it) == len(self.tn_mc)
    def append_train_malis_eng( self, me ):
        while len( self.tn_me ) < len( self.tn_it )-1:
            self.tn_me.append( np.nan )
        self.tn_me.append( me )

    def _append_test_rand_error( self, re ):
        while len( self.tt_re ) < len(self.tt_it)-1:
            self.tt_re.append( np.nan )
        self.tt_re.append( re )
        assert len(self.tt_it) == len(self.tt_re)

    def append_test_malis_cls( self, mc ):
        while len( self.tt_mc ) < len(self.tt_it)-1:
            self.tt_mc.append( np.nan )
        self.tt_mc.append( mc )
        assert len(self.tt_it) == len(self.tt_mc)

    def append_test_malis_eng( self, me ):
        while len( self.tt_me ) < len( self.tt_it )-1:
            self.tt_me.append( np.nan )
        self.tt_me.append( me )

    def get_last_it(self):
        # return the last iteration number
        if len(self.tt_it)>0 and len(self.tn_it)>0:
            last_it = max( self.tt_it[-1], self.tn_it[-1] )
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
        print "max iter: ", self._find_max_update( self.tn_it, self.tn_cls )

    def show(self, w):
        """
        illustrate the learning curve

        Parameters
        ----------
        w : int, window size for smoothing the curve
        """
        if len(self.tn_mc) > 0:
            # malis training, increase number of subplots
            nsp = 5
        else:
            nsp = 3

        # print the maximum iteration
        self.print_max_update()

        # using K as iteration unit
        tn_it = self.tn_it
        for i in xrange(len(tn_it)):
            tn_it[i] = tn_it[i] / float(1000)
        tt_it = self.tt_it
        for i in xrange(len(tt_it)):
            tt_it[i] = tt_it[i] / float(1000)

        # plot data
        plt.subplot(1,nsp, 1)
        plt.plot(tn_it, self.tn_err, 'b.', alpha=0.2)
        plt.plot(tt_it, self.tt_err, 'r.', alpha=0.2)
        # plot smoothed line
        xne,yne = self._smooth( tn_it, self.tn_err, w )
        xte,yte = self._smooth( tt_it, self.tt_err, w )
        plt.plot(xne, yne, 'b')
        plt.plot(xte, yte, 'r')
        plt.xlabel('iteration (K)'), plt.ylabel('cost energy')

        plt.subplot(1,nsp,2)
        plt.plot(tn_it, self.tn_cls, 'b.', alpha=0.2)
        plt.plot(tt_it, self.tt_cls, 'r.', alpha=0.2)
        # plot smoothed line
        xnc, ync = self._smooth( tn_it, self.tn_cls, w )
        xtc, ytc = self._smooth( tt_it, self.tt_cls, w )
        plt.plot(xnc, ync, 'b', label='train')
        plt.plot(xtc, ytc, 'r', label='test')
        plt.xlabel('iteration (K)'), plt.ylabel( 'classification error' )

        if len(tn_it) == len( self.tn_re ):
            plt.subplot(1, nsp, 3)
            plt.plot(tn_it, self.tn_re, 'b.', alpha=0.2)
            plt.plot(tt_it, self.tt_re, 'r.', alpha=0.2)
            # plot smoothed line
            xnr, ynr = self._smooth( tn_it, self.tn_re, w )
            xtr, ytr = self._smooth( tt_it, self.tt_re, w )
            plt.plot(xnr, ynr, 'b', label='train')
            plt.plot(xtr, ytr, 'r', label='test')
            plt.xlabel('iteration (K)'), plt.ylabel( 'rand error' )


        if len(tn_it) == len( self.tn_mc ):
            plt.subplot(1, nsp, 4)
            plt.plot(tn_it, self.tn_mc, 'b.', alpha=0.2)
            plt.plot(tt_it, self.tt_mc, 'r.', alpha=0.2)
            # plot smoothed line
            xnm, ynm = self._smooth( tn_it, self.tn_mc, w )
            xtm, ytm = self._smooth( tt_it, self.tt_mc, w )
            plt.plot(xnm, ynm, 'b', label='train')
            plt.plot(xtm, ytm, 'r', label='test')
            plt.xlabel('iteration (K)'), plt.ylabel( 'malis weighted cost energy' )

        if len(tn_it) == len( self.tn_me ):
            plt.subplot(1, nsp, 5)
            plt.plot(tn_it, self.tn_me, 'b.', alpha=0.2)
            plt.plot(tt_it, self.tt_me, 'r.', alpha=0.2)
            # plot smoothed line
            xng, yng = self._smooth( tn_it, self.tn_me, w )
            xtg, ytg = self._smooth( tt_it, self.tt_me, w )
            plt.plot(xng, yng, 'b', label='train')
            plt.plot(xtg, ytg, 'r', label='test')
            plt.xlabel('iteration (K)'), plt.ylabel( 'malis weighted pixel error' )

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

            if len(self.tn_it) > 0:
                fname = root + '_statistics_{}.h5'.format( self.tn_it[-1] )
            else:
                fname = root + '_statistics_0.h5'
            if os.path.exists(fname):
                os.remove( fname )

        # save variables
        import h5py
        f = h5py.File( fname, 'a' )
        f.create_dataset(self.stdpre + '/train/it',  data=self.tn_it )
        f.create_dataset(self.stdpre + '/train/err', data=self.tn_err)
        f.create_dataset(self.stdpre + '/train/cls', data=self.tn_cls)
        if pars['is_malis'] :
            f.create_dataset(self.stdpre + '/train/re',  data=self.tn_re )
            f.create_dataset(self.stdpre + '/train/mc',  data=self.tn_mc )
            f.create_dataset(self.stdpre + '/train/me',  data=self.tn_me )

        f.create_dataset(self.stdpre + '/test/it',   data=self.tt_it )
        f.create_dataset(self.stdpre + '/test/err',  data=self.tt_err)
        f.create_dataset(self.stdpre + '/test/cls',  data=self.tt_cls)
        if pars['is_malis'] :
            f.create_dataset(self.stdpre + '/test/re',   data=self.tt_re )
            f.create_dataset(self.stdpre + '/test/mc',   data=self.tt_mc )
            f.create_dataset(self.stdpre + '/test/me',   data=self.tt_me )

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

    fconfig = path.dirname(fname) + "/config.cfg"
    from front_end import zconfig

    lc = CLearnCurve( fname )

    if len(sys.argv)==3:
        w = int( sys.argv[2] )
        print "window size: ", w
    lc.show( w )
