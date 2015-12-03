# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 14:28:47 2015

@author: jingpeng
"""
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors

class CompareVol:
    '''A plotting object which encapsulates functionality
    described in compare_volumes.py'''
    def __init__(self, vols, cmap='gray'):

        #zero-padded copies of the volumes
        self.vols = self.__pad(vols)
        #Number of slices to display
        self.Nz = min([elem.shape[0]-1 for elem in vols])
        #Current z index
        self.z = 0
        #Colormap argument to be passed to imshow (set under 'c' keypress)
        self.cmap = [cmap for elem in vols]
        #Whether or not the plot at an index is a color plot
        self.colorplot = [False for elem in vols]

        #Defining a current index of the vols to be 'selected' for
        # modification by (for example) adding color
        self.selected = 0

    def __pad(self, vols):
        '''Zero-padding a list of input volumes to match by non-z dimensions'''
        shapes = np.array(
            [elem.shape for elem in vols]
            )

        max_shape = np.max(shapes,0)

        pad_vols = [np.zeros((elem.shape[0], max_shape[1], max_shape[2]))
                    for elem in vols]

        dim_diffs = [(max_shape - elem.shape) / 2
                    for elem in vols]

        for i in range(len(pad_vols)):

            if all(dim_diffs[i][1:] != 0):
                pad_vols[i][
                    :,
                    dim_diffs[i][1]:-(dim_diffs[i][1]),
                    dim_diffs[i][2]:-(dim_diffs[i][2])
                    ] = vols[i]
            else:
                pad_vols[i] = vols[i]

        return pad_vols

    def __norm(self, imslice):
        #subtract the nonzero minimum from each slice
        nonzero_indices = np.nonzero(imslice)
        if len(nonzero_indices) > 0 and np.max(imslice) > 1:
            nonzero_min = np.min(imslice[np.nonzero(imslice)])

            res = np.copy(imslice)
            res[np.nonzero(res)] = res[np.nonzero(res)] - nonzero_min + 1
        else:
            res = imslice
        return res

    def __show_slice(self):
        '''Basic display function'''

        for i in range(1,len(self.axs)+1):
            ax = self.axs[i-1]
            ax.images.pop()
            normed_slice = self.__norm(self.vols[i-1][self.z,:,:])
            ax.imshow(normed_slice, interpolation='nearest', cmap=self.cmap[i-1])
            ax.set_xlabel( ' volume {}: slice {}'.format(i,self.z) )

        self.fig.canvas.draw()

    def __make_cmap(self, i):

        #(0,0,0) = black
        plot_colors = np.vstack(((0,0,0), np.random.rand(500,3)))
        cmap = colors.ListedColormap(plot_colors)


        return cmap

    def __press(self, event):
#       print 'press ' + event.key
        if 'down' == event.key and self.z<self.Nz:
            self.z+=1
        elif 'up' == event.key and self.z>-self.Nz:
            self.z-=1
        elif 'c' == event.key:
            #Swap between color display and b&w
            self.colorplot[self.selected] = not self.colorplot[self.selected]

            if self.colorplot[self.selected]:
                new_cmap = self.__make_cmap(self.selected)

                self.cmap[self.selected] = new_cmap

            else:
                self.cmap[self.selected] = 'gray'

        elif 'j' == event.key:
            self.z += 10

        elif 'k' == event.key:
            self.z -= 10

        elif 'v' == event.key:
            #Display the data values for the given data coordinate
            xcoord, ycoord = int(event.xdata), int(event.ydata)
            print xcoord, ycoord

            print [vol[self.z, ycoord, xcoord] for vol in self.vols]


        elif event.key in ['1','2','3','4','5','6','7','8','9']:
            #Select a new axis
            index = int(event.key)

            if index - 1 < len(self.vols):
                self.selected = index-1

        self.__show_slice()

    def vol_compare_slice(self):
        self.fig, self.axs = plt.subplots(1,len(self.vols), sharex=True, sharey=True)
        self.fig.canvas.mpl_connect('key_press_event', self.__press)

        for i in range(1,len(self.vols)+1):
            ax = self.axs[i-1]
            normed_slice = self.__norm(self.vols[i-1][self.z,:,:])
            ax.imshow(normed_slice, interpolation='nearest', cmap=self.cmap[i-1])
            ax.set_xlabel( ' volume {0}: slice {1}'.format(i,self.z) )
        plt.show()

def compare_volumes( vols  ):
    """
    compare 3D volumes

    Parameters
    ----------
    vols: list of 3D array
    """
    com = CompareVol(vols)
    com.vol_compare_slice()
    return

class VolSlider:
    def __init__(self, fname, cmap='gray'):
        if ".h5" in fname or ".hdf5" in fname:
            import h5py
            f = h5py.File(fname)
            self.v1 = f['/main']
            if len(f['/main'].dims) == 3:
                self.Nz = self.v1.dims[0]
            elif len(f['/main'].dims) == 4:
                self.Nz = self.v1.dims[1]
            else:
                raise RuntimeError('invalid input matrix dimension')
        else:
            import emirt.io
            self.v1 = emirt.io.imread( fname )
            self.Nz = self.v1.shape[0]
        self.z = 0
        self.cmap = cmap

    def __show_slice(self):
        self.ax1.images.pop()
        if len(self.v1.dims) == 3:
            self.ax1.imshow(self.v1[self.z,:,:], interpolation='nearest', cmap=self.cmap)
        else:
            self.ax1.imshow(self.v1[0,self.z,:,:], interpolation='nearest', cmap=self.cmap)
        self.ax1.set_xlabel( 'first volume: slice {}'.format(self.z) )
        self.fig.canvas.draw()

    def __press(self, event):
#        print 'press ' + event.key
        if 'down' in event.key and self.z<self.Nz:
            self.z+=1
        elif 'up' in event.key and self.z>-self.Nz:
            self.z-=1
        self.__show_slice()

    def show(self):
        self.fig, self.ax1 = plt.subplots(1,1)
        self.fig.canvas.mpl_connect('key_press_event', self.__press)
        if len(self.v1.dims) == 3:
            self.ax1.imshow(self.v1[self.z,:,:], interpolation='nearest', cmap=self.cmap)
        else:
            self.ax1.imshow(self.v1[0,self.z,:,:], interpolation='nearest', cmap=self.cmap)
        self.ax1.set_xlabel( 'first  volume: slice {}'.format(self.z) )

def matshow(mat, xlabel=''):
    import matplotlib.pylab as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.matshow(mat, cmap=plt.cm.gray_r)
    # add numbers
    Nx, Ny = mat.shape
    x,y = np.meshgrid(range(Nx), range(Ny))
    for i,j in zip(x.ravel(),y.ravel()):
        s = str( np.round(mat[i,j], decimals=2) )
        if mat[i,j]<np.mean(mat):
            ax1.annotate(s, xy=(i,j), ha='center', va='center')
        else:
            ax1.annotate(s, xy=(i,j), ha='center', va='center', color='white')
    ax1.set_xlabel(xlabel)
    plt.show()

def imshow(im):
    import matplotlib.pylab as plt
    plt.imshow(im)

# show the labeled image with random color
def random_color_show( im, mode='mat' ):
    import matplotlib.pylab as plt
    import matplotlib.colors as mcolor
    # make a random color map, but the background should be black
    assert(im.max()>0)
    cmap_array = np.random.rand ( im.max()+1,3)
    cmap_array[0,:] = [0,0,0]
    cmap=mcolor.ListedColormap( cmap_array )
    if mode=='im':
        plt.imshow(im, cmap= cmap )
    elif mode=='mat':
        # approximate the matshow for compatability of subplot
        nr, nc = im.shape
        extent = [-0.5, nc-0.5, nr-0.5, -0.5]
        plt.imshow(im, extent=extent, origin='upper',interpolation='nearest', cmap=cmap)
#        plt.matshow(im, cmap=mcolor.ListedColormap( cmap_array ) )
    else:
        print 'unknown mode'
    plt.show()

def progress(count, total, suffix=''):
    import sys
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
