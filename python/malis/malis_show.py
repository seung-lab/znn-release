import emirt
import numpy as np
import matplotlib.pylab as plt

def compute_gradient(bdm, lbl, me, se):
    # gradient
    # square loss gradient
    grdt = 2 * (bdm-  (lbl>0).astype('float32'))
    # merger and splitter gradient
    mg = grdt * me
    sg = grdt * se
    return mg, sg

# rescale to 0-1
def rescale(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr

def combine2rgb(bdm, w=None):
    # make a combined colorful image
    cim = np.zeros(bdm.shape+(3,), dtype='uint8')
    # red channel
    cim[:,:,0] = (rescale(bdm))*255
    # green channel
    if w is not None:
        cim[:,:,1] = rescale( w )*255
    return cim


def disk_plot(e, D, DrTh, color='g'):
    # plot disk to illustrate the weight strength
    if np.all(e==0):
        return
    # rescale to 0-1
    re = rescale(e) * D
    y,x = np.nonzero(re)
    r = re[(y,x)]
    # sort the disk from small to large
    locs = np.argsort( r )
    y = y[ locs ]
    x = x[ locs ]
    r = r[ locs ]
    # eleminate the small disks
    y = y[ r > DrTh ]
    x = x[ r > DrTh ]
    r = r[ r > DrTh ]
    plt.scatter(x,y,r, c=color, alpha=0.6, linewidths=0)

    # print the maximum value
    dev = max(x.max(), y.max()) * 0.07
    plt.annotate("%d" % e.max(), xy=(x[-1],y[-1]),
                xytext=(x[-1]+dev, y[-1]+dev),
                color = 'white',
                arrowprops=dict(color='white',
                arrowstyle="->"))


def plot_aff( pars, affs, lbl, me, se):
    # parameters
    eps = pars['eps']
    Dm = pars['Dm']
    Ds = pars['Ds']
    DrTh = pars['DrTh']

    # the values
    zaff = affs[0,0,:,:]
    yaff = affs[1,0,:,:]
    xaff = affs[2,0,:,:]

    zme = me[0,0,:,:]
    yme = me[1,0,:,:]
    xme = me[2,0,:,:]

    zse = se[0,0,:,:]
    yse = se[1,0,:,:]
    xse = se[2,0,:,:]

    # 6th column show true affinity

    # 1st column show affinity
    plt.subplot(351)
    plt.imshow(xaff, cmap='gray', interpolation='nearest')
    plt.xlabel('x affinity')

    plt.subplot(356)
    plt.imshow(yaff, cmap='gray', interpolation='nearest')
    plt.xlabel('y affinity')

    plt.subplot(3,5,11)
    plt.imshow(zaff, cmap='gray', interpolation='nearest')
    plt.xlabel('z affinity')

    # 2nd column show merge error in log scale
    rgbmxl = combine2rgb( 1-xaff, np.log(xme+eps) )
    plt.subplot(352)
    plt.imshow( rgbmxl, interpolation='nearest' )
    plt.xlabel('x affinity (red) and ln(merger)')

    rgbmyl = combine2rgb( 1-yaff, np.log(yme+eps) )
    plt.subplot(357)
    plt.imshow( rgbmyl, interpolation='nearest' )
    plt.xlabel('y affinity (red) and ln(merger)')

    rgbmzl = combine2rgb( 1-zaff, np.log(zme+eps) )
    plt.subplot(3,5,12)
    plt.imshow( rgbmzl, interpolation='nearest' )
    plt.xlabel('z affinity (red) and ln(merger)')

    # 3rd column show split error in log scale
    rgbsxl = combine2rgb( 1-xaff, np.log(xse+eps) )
    plt.subplot(353)
    plt.imshow( rgbsxl, interpolation='nearest' )
    plt.xlabel('x affinity (red) and ln(splitter)')

    rgbsyl = combine2rgb( 1-yaff, np.log(yse+eps) )
    plt.subplot(358)
    plt.imshow( rgbsyl, interpolation='nearest' )
    plt.xlabel('y affinity (red) and ln(splitter)')

    rgbszl = combine2rgb( 1-zaff, np.log(zse+eps) )
    plt.subplot(3,5,13)
    plt.imshow( rgbszl, interpolation='nearest' )
    plt.xlabel('z affinity (red) and ln(splitter)')

    # 4th column show merger error using disks
    rgbx = combine2rgb( 1-xaff )
    rgby = combine2rgb( 1-yaff )
    rgbz = combine2rgb( 1-zaff )

    plt.subplot(3,5,4)
    plt.imshow(rgbx, interpolation='nearest')
    disk_plot(xme, Dm, DrTh)
    plt.xlabel('combine xaff and merger weights')

    plt.subplot(3,5,9)
    plt.imshow(rgby, interpolation='nearest')
    disk_plot(yme, Dm, DrTh)
    plt.xlabel('combine yaff and merger weights')

    plt.subplot(3,5,14)
    plt.imshow(rgbz, interpolation='nearest')
    disk_plot(zme, Dm, DrTh)
    plt.xlabel('combine zaff and merger weights')

    # 5th column show split error using disks
    plt.subplot(355)
    plt.imshow(rgbx, interpolation='nearest')
    disk_plot(xse, Dm, DrTh)
    plt.xlabel('combine xaff and splitter weights')

    plt.subplot(3,5,10)
    plt.imshow(rgby, interpolation='nearest')
    disk_plot(yse, Dm, DrTh)
    plt.xlabel('combine yaff and splitter weights')

    plt.subplot(3,5,15)
    plt.imshow(rgbz, interpolation='nearest')
    disk_plot(zse, Dm, DrTh)
    plt.xlabel('combine zaff and splitter weights')

    plt.show()

def plot_bdm( pars, bdm, lbl, me, se, mbdm=None, sbdm=None):
    is_constrained = pars['is_constrained']
    eps = pars['eps']
    Dm = pars['Dm']
    Ds = pars['Ds']
    DrTh = pars['DrTh']

    #%% plot the results
    print "plot the images"
    if is_constrained:
        mbdm = np.copy(bdm)
        mbdm[lbl>0] = 1
        plt.subplot(241)
        plt.imshow(mbdm, cmap='gray', interpolation='nearest')
        plt.xlabel('merger constrained boundary map')

        sbdm = np.copy(bdm)
        sbdm[lbl==0] = 0
        plt.subplot(245)
        plt.imshow(sbdm, cmap='gray', interpolation='nearest')
        plt.xlabel('splitter constrained boundary map')
    else:
        plt.subplot(241)
        plt.imshow(bdm, cmap='gray', interpolation='nearest')
        plt.xlabel('boundary map')
        plt.subplot(245)
        plt.imshow(lbl>0, cmap='gray', interpolation='nearest')
        #emirt.show.random_color_show( lbl, mode='mat' )
        plt.xlabel('manual labeling')

    # combine merging error with boundary map
    rgbm = combine2rgb(1-bdm, np.log(me+eps))
    plt.subplot(242)
    plt.imshow( rgbm, interpolation='nearest' )
    plt.xlabel('combine boundary map(red) and ln(merge weight)(green)')

    # combine merging error with boundary map
    rgbs = combine2rgb(1-bdm, np.log(se+eps))
    plt.subplot(246)
    plt.imshow( rgbs, interpolation='nearest' )
    plt.xlabel('combine boundary map(red) and ln(split weight)(green)')

    # combine merging error with boundary map
    rgb_bdm = combine2rgb(1-bdm)
    plt.subplot(243)
    plt.imshow( rgb_bdm, interpolation='nearest' )
    plt.xlabel('combine boundary map(red) \n and merge weight(green disk)')
    disk_plot(me, Dm, DrTh)

    plt.subplot(247)
    plt.imshow( rgb_bdm, interpolation='nearest' )
    plt.xlabel('combine boundary map(red) \n and split weight(green disk)')
    disk_plot(se, Ds, DrTh)

    # merger and spliter gradient
    mg, sg = compute_gradient(bdm, lbl, me, se)

    plt.subplot(244)
    #cim,mpg,mng = gradient2rgb(mg)
    mgcim = combine2rgb(1-bdm)
    plt.imshow(mgcim, interpolation='nearest')
    disk_plot( np.abs(mg), Dm, DrTh, color='g')
    plt.xlabel('merger gradient (square loss absolute value)')


    plt.subplot(248)
    #cim,spg,sng = gradient2rgb(sg)
    sgcim = combine2rgb(1-bdm)
    plt.imshow(sgcim, interpolation='nearest')
    disk_plot( np.abs(sg), Ds, DrTh, color='g' )
    plt.xlabel('splitter gradient (square loss absolute value)')

    plt.show()

def plot(pars, data, lbl, me, se):
    if pars['is_affinity']:
        plot_aff( pars, data, lbl, me, se )
    else:
        plot_bdm( pars, data, lbl, me, se )

def read_bin_img(fname, shape=(504,504)):
    """
    read binary image file as metrics
    """
    img = np.fromfile(fname, dtype='float64')
    img = img.astype('float32').reshape(shape)
    return img

if __name__ == '__main__':
    from malis_test import get_params
    pars = get_params()
    is_constrained = pars['is_constrained']
    if is_constrained:
        mbdm = read_bin_img("../dataset/malis/bdm_merge.bin")
        sbdm = read_bin_img("../dataset/malis/bdm_splite.bin")

    bdm = read_bin_img("../dataset/malis/boundary_map.bin")
    lbl = read_bin_img("../dataset/malis/label.bin")

    me  = read_bin_img("../experiments/malis/out_constrained.merger",   (504,504,2))
    se  = read_bin_img("../experiments/malis/out_constrained.splitter", (504,504,2))
    me = me[:,:,0] + me[:,:,1]
    se = se[:,:,0] + se[:,:,1]

    if is_constrained:
        plot( lbl, me, se, pars, bdm, mbdm, sbdm)
    else:
        plot( lbl, me, se, pars, bdm)
