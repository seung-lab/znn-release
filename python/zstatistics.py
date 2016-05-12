#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import time
import os
import utils

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

        assert( os.path.exists(fname) )
        # read data
        import h5py
        # read into memory
        f = h5py.File(fname, 'r', driver='core')

        if "/processing/znn/train/statistics/" in f:
            self.stdpre = "/processing/znn/train/statistics/"
        else:
            self.stdpre = "/"

        print "stdpre: ", self.stdpre
        ft = f[self.stdpre + 'test']
        for key in ft:
            self.test[key] = list(ft[key].value)

        ft = f[self.stdpre + 'train/']
        for  key in ft:
            self.train[key] = list(ft[key].value)

        f.close()

        # crop the values
        if iter_num is not None:
            self._crop_iters(iter_num)
        return

    # fetch the latest parameter
    def fetch(self, key):
        if self.train.has_key(key):
            return self.train[key][-1]
        else:
            return None

    # push an item to training
    def push(self, key, value):
        if self.train.has_key(key):
            self.train[key].append(value)
        else:
            self.train[key] = [value]

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
        root, ext = os.path.splitext(fname)
        str_num = root.split('_')[-1]
        if 'current' in str_num :
            # the last network
            return None
        else:
            return int(str_num)

    def append_test(self, derr):
        # add the iter first
        if not self.test.has_key('it'):
            self.test['it'] = [ derr['it'] ]
        else:
            self.test['it'].append(derr['it'])
        # add a test result
        for key in derr.keys():
            if key=='it':
                continue
            if not self.test.has_key(key):
                self.test[key] = list()
            while len( self.test[key] ) < len(self.test['it'])-1:
                # fill with nan
                self.test[key].append( np.nan )
            self.test[key].append(derr[key])

    def append_train(self, history):
        # add the iter first
        if not self.train.has_key('it'):
            self.train['it'] = [ history['it'] ]
        else:
            self.train['it'].append(history['it'])
        # add a test result
        for key in history.keys():
            if key=='it':
                continue
            if not self.train.has_key(key):
                self.train[key] = list()
            while len( self.train[key] ) < len(self.train['it'])-1:
                # fill with nan
                self.train[key].append( np.nan )
            self.train[key].append(history[key])

    # get learning rate
    def get_last_eta(self):
        if self.train.has_key('eta'):
            return self.train['eta'][-1]
        else:
            return None

    def get_last_it(self):
        # return the last iteration number
        if self.train.has_key('it') and len(self.test['it'])>0 and len(self.train['it'])>0:
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

    def plot(self, w=3, plotmode='matplotlib'):
        """
        illustrate the learning curve

        Parameters
        ----------
        w : int, window size for smoothing the curve
        plotmode: string, options: matplotlib | plotly
        """
        import matplotlib.pylab as plt
        fig = plt.figure()

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


        # dictionary of key and meaning
        kd = {'err':'cost', 'cls':'pixel error', 'mc':'malis cost', 'me':'malis error','re':'rand error'}
        # plot data
        idx = 0
        for key in self.train.keys():
            if key == 'it':
                continue
            idx += 1
            ax = fig.add_subplot(1,nsp,idx)
            ax.plot(self.train['it'], self.train[key], 'b.', alpha=0.2)
            ax.plot(self.test['it'],  self.test[key],  'r.', alpha=0.2)
            # plot smoothed line
            xne,yne = self._smooth( self.train['it'], self.train[key], w )
            xte,yte = self._smooth( self.test['it'],  self.test[key],  w )
            ax.plot(xne, yne, 'b')
            ax.plot(xte, yte, 'r')
            #ax.set_
            ax.set_xlabel('iteration (K)')
            ax.set_ylabel(kd[key])

        plt.tight_layout()
        plt.legend()
        fig = plt.gcf()
        if 'matplot' in plotmode:
            plt.show()
        elif 'plotly' in plotmode:
            # transform to plotly fig
            import plotly.plotly as py
            import plotly.tools as tls
            plotly_fig = tls.mpl_to_plotly( fig )
            plotly_fig['layout']['title'] = 'Learning Curve'
            plotly_fig['layout']['margin'].update({'t':40})
            plot_url = py.plot(plotly_fig, filename='znn-learning-curve', validate=False)
            print plot_url
        else:
            raise NameError("invalid plot mode! should be either matplotlib or plotly")
        return

    def save(self, pars, fname=None, suffix=None):
        if pars['is_stdio']:
            self.stdpre = "/processing/znn/train/statistics/"
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

        for key, value in self.train.iteritems():
            f.create_dataset("{}train/{}".format(self.stdpre, key),  data=value )

        for key, value in self.test.iteritems():
            f.create_dataset(self.stdpre + "test/" + key,  data=np.asarray(value) )

        f.close()

        if not pars['is_stdio']:
            # move to new name
            fname2 = root + '_statistics_current.h5'
            if os.path.exists( fname2 ):
                os.remove( fname2 )
            shutil.copyfile(fname, fname2)

def init_history(pars, lc):
    # initialization
    history = dict()
    history['elapse'] = 0
    history['err'] = 0.0 # cost energy
    history['cls'] = 0.0 # pixel classification error
    history['re'] = 0.0  # rand error
    # number of voxels which accumulate error
    # (if a mask exists)
    history['num_mask_voxels'] = 0

    if pars['is_malis']:
        # malis cost
        history['mc'] = 0.0
        # malis error
        history['me'] = 0.0
        # malis weight
        # history['mw'] = 0.0
    # load existing learning rate
    if pars['eta']<=0 and lc.get_last_eta():
        history['eta'] = lc.get_last_eta()
    else:
        history['eta'] = pars['eta']

    return history

def show_history(history):
    if history.has_key('mc'):
        show_string = "update %d,    cost: %.3f, pixel error: %.3f, me: %.3f, mc: %.3f, elapse: %.2f s/iter, learning rate: %.5f" %(history['it'], history['err'], history['cls'],  history['me'], hostory['mc'], history['elapse'], history['eta'] )
    else:
        show_string = "update %d,    cost: %.3f, pixel error: %.3f, elapse: %.2f s/iter, learning rate: %.5f" %(history['it'], history['err'], history['cls'], history['elapse'], history['eta'] )
    print show_string

def reset_history(history):
    for key in history.keys():
        if not 'eta' in key:
            history[key] = 0
    return history

def process_history(pars, history, lc, net, it, start, total_time):
    if it%pars['Num_iter_per_show']==0:
        # iter number
        history['it'] = it
        # time
        history['elapse'] = total_time / pars['Num_iter_per_show']
        # reset the start time
        total_time = 0.0
        start = time.time()

        # normalize
        history['err'] /= pars['Num_iter_per_show']
        history['cls'] /= pars['Num_iter_per_show']
        if pars['is_malis']:
            history['mc'] /= pars['Num_iter_per_show']
            history['me'] /= pars['Num_iter_per_show']

        lc.append_train(history)

        show_history(history)

        if pars.has_key('logging') and pars['logging']:
            utils.write_to_log(logfile, show_string)

        # reset history
        history = reset_history(history)

    if it%pars['Num_iter_per_annealing']==0:
        # anneal factor
        history['eta'] *= pars['anneal_factor']
        net.set_eta(history['eta'])

    return history, net, lc, start, total_time


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

    if len(sys.argv) >2:
        w = int( sys.argv[2] )
        print "window size: ", w
    else:
        w = 3
    if len(sys.argv) >3:
        plotmode = sys.argv[3]
    else:
        plotmode = 'matplotlib'
    lc.plot( w, plotmode )
