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

        if 'statistics' not in fname:
            # it is the network file name
            fname = find_statistics_file_within_dir(fname)
        assert( path.exists(fname) )
        # read data
        import h5py
        # read into memory
        f = h5py.File(fname, 'r', driver='core')
        self.tt_it  = list( f['/test/it'].value )
        self.tt_err = list( f['/test/err'].value )
        self.tt_cls = list( f['/test/cls'].value )
        if '/test/re' in f:
            self.tt_re = list( f['/test/re'].value )
        else:
            self.tt_re = list()
        if '/test/mc' in f:
            self.tt_mc = list( f['/test/mc'].value )
        else:
            self.tt_mc = list()
        if '/test/me' in f:
            self.tt_me = list( f['/test/me'].value )
        else:
            self.tt_me = list()

        self.tn_it  = list( f['/train/it'].value )
        self.tn_err = list( f['/train/err'].value )
        self.tn_cls = list( f['/train/cls'].value )
        if '/train/re' in f:
            self.tn_re = list( f['/train/re'].value )
        else:
            self.tn_re = list()

        if '/train/mc' in f:
            self.tn_mc = list( f['/train/mc'].value )
        else:
            self.tn_mc = list()

        if '/train/me' in f:
            self.tn_me = list( f['/train/me'].value )
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
        if 'current' in str_num or 'statistics' in str_num:
            # the last network
            return None
        else:
            return int(str_num)

    def append_test(self, it, err, cls):
        # add a test result
        self.tt_it.append(it)
        self.tt_err.append(err)
        self.tt_cls.append(cls)

    def append_train(self, it, err, cls):
        # add a training result
        self.tn_it.append(it)
        self.tn_err.append(err)
        self.tn_cls.append(cls)

    def append_train_rand_error( self, re ):
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

    def append_test_rand_error( self, re ):
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
        # plot data
        plt.subplot(1,nsp, 1)
        plt.plot(self.tn_it, self.tn_err, 'b.', alpha=0.2)
        plt.plot(self.tt_it, self.tt_err, 'r.', alpha=0.2)
        # plot smoothed line
        xne,yne = self._smooth( self.tn_it, self.tn_err, w )
        xte,yte = self._smooth( self.tt_it, self.tt_err, w )
        plt.plot(xne, yne, 'b')
        plt.plot(xte, yte, 'r')
        plt.xlabel('iteration'), plt.ylabel('cost energy')

        plt.subplot(1,nsp,2)
        plt.plot(self.tn_it, self.tn_cls, 'b.', alpha=0.2)
        plt.plot(self.tt_it, self.tt_cls, 'r.', alpha=0.2)
        # plot smoothed line
        xnc, ync = self._smooth( self.tn_it, self.tn_cls, w )
        xtc, ytc = self._smooth( self.tt_it, self.tt_cls, w )
        plt.plot(xnc, ync, 'b', label='train')
        plt.plot(xtc, ytc, 'r', label='test')
        plt.xlabel('iteration'), plt.ylabel( 'classification error' )

        if len(self.tn_it) == len( self.tn_re ):
            plt.subplot(1, nsp, 3)
            plt.plot(self.tn_it, self.tn_re, 'b.', alpha=0.2)
            plt.plot(self.tt_it, self.tt_re, 'r.', alpha=0.2)
            # plot smoothed line
            xnr, ynr = self._smooth( self.tn_it, self.tn_re, w )
            xtr, ytr = self._smooth( self.tt_it, self.tt_re, w )
            plt.plot(xnr, ynr, 'b', label='train')
            plt.plot(xtr, ytr, 'r', label='test')
            plt.xlabel('iteration'), plt.ylabel( 'rand error' )

        if len(self.tn_it) == len( self.tn_mc ):
            plt.subplot(1, nsp, 4)
            plt.plot(self.tn_it, self.tn_mc, 'b.', alpha=0.2)
            plt.plot(self.tt_it, self.tt_mc, 'r.', alpha=0.2)
            # plot smoothed line
            xnm, ynm = self._smooth( self.tn_it, self.tn_mc, w )
            xtm, ytm = self._smooth( self.tt_it, self.tt_mc, w )
            plt.plot(xnm, ynm, 'b', label='train')
            plt.plot(xtm, ytm, 'r', label='test')
            plt.xlabel('iteration'), plt.ylabel( 'malis weighted pixel \n classification error' )

        if len(self.tn_it) == len( self.tn_me ):
            plt.subplot(1, nsp, 4)
            plt.plot(self.tn_it, self.tn_me, 'b.', alpha=0.2)
            plt.plot(self.tt_it, self.tt_me, 'r.', alpha=0.2)
            # plot smoothed line
            xng, yng = self._smooth( self.tn_it, self.tn_me, w )
            xtg, ytg = self._smooth( self.tt_it, self.tt_me, w )
            plt.plot(xng, yng, 'b', label='train')
            plt.plot(xtg, ytg, 'r', label='test')
            plt.xlabel('iteration'), plt.ylabel( 'malis weighted cost energy' )


        plt.legend()
        plt.show()
        return

    def save(self, pars, elapsed):
        # get filename
        fname = pars['train_save_net']
        import os
        import shutil
        root, ext = os.path.splitext(fname)
        fname = root + '_statistics_{}.h5'.format( self.tn_it[-1] )
        if os.path.exists(fname):
            os.remove( fname )

        # save variables
        import h5py
        f = h5py.File( fname )
        f.create_dataset('/train/it',  data=self.tn_it )
        f.create_dataset('/train/err', data=self.tn_err)
        f.create_dataset('/train/cls', data=self.tn_cls)
        f.create_dataset('/train/re',  data=self.tn_re )
        f.create_dataset('/train/mc',  data=self.tn_mc )
        f.create_dataset('/train/me',  data=self.tn_me )

        f.create_dataset('/test/it',   data=self.tt_it )
        f.create_dataset('/test/err',  data=self.tt_err)
        f.create_dataset('/test/cls',  data=self.tt_cls)
        f.create_dataset('/test/re',   data=self.tt_re )
        f.create_dataset('/test/mc',   data=self.tt_mc )
        f.create_dataset('/test/me',   data=self.tt_me )

        f.create_dataset('/elapsed',   data=elapsed)
        f.close()

        # move to new name
        fname2 = root + '_statistics_current.h5'
        if os.path.exists( fname2 ):
            os.remove( fname2 )
        shutil.copyfile(fname, fname2)

def find_statistics_file_within_dir(seed_filename):
    '''
    Looks for the stats file amongst the directory where
    the loaded network is stored
    '''
    import glob

    containing_directory, filename = path.split(seed_filename)

    #First attempt- if there's only one stats file, take it
    candidate_files = glob.glob( "{}/*statistics*".format(containing_directory) )

    some_stats_files_in_load_directory = len(candidate_files) > 0
    assert(some_stats_files_in_load_directory)

    #Next attempt- split filename by '_' and search for more specific files
    # until only one remains

    filename_fields = filename.split('_')
    filename_fields.reverse()

    first_field = filename_fields.pop()
    search_expression_head = containing_directory + "/" + first_field

    exact_file = search_expression_head + "_statistics_" + filename_fields.pop()
    import os
    if os.path.exists( exact_file ):
        # have one statistics file matches exactly!
        return exact_file

    while len(candidate_files) > 1:
        candidate_files = glob.glob( search_expression_head + "*statistics*" )

        stats_search_found_a_file = len(candidate_files) > 0
        assert(stats_search_found_a_file)

        search_expression_head += '_' + filename_fields.pop()

    return candidate_files[0]

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

    if len(sys.argv)==2:
        fname = sys.argv[1]
        lc = CLearnCurve( fname )
    elif len(sys.argv)==3:
        fname = sys.argv[1]
        lc = CLearnCurve( fname )
        w = int( sys.argv[2] )
        print "window size: ", w
    else:
        raise NameError("no input statistics h5 file!")

    lc.show( w )
