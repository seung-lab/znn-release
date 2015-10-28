# test the malis of boundary map
import emirt
import numpy as np
import time
#%% parameters
z = 8
# epsilone: a small number for log to avoind -infinity
eps = 0.0000001

# largest diameter
Dm = 100
Ds = 50

# whether using constrained malis
is_constrained = True

#%% read images

bdm = emirt.emio.imread('../experiments/zfish/VD2D/out_sample91_output_0.tif')
lbl = emirt.emio.imread('../dataset/zfish/Merlin_label2_24bit.tif')
lbl = emirt.volume_util.lbl_RGB2uint32(lbl)
lbl = lbl[z,:,:]
bdm = bdm[z,:,:]


import cost_fn
start = time.time()
if is_constrained:
    print "compute the constrained malis weight..."
    w, me, se = cost_fn.constrained_malis_weight_bdm_2D(bdm, lbl)
else:
    print "compute the normal malis weight..."
    w, me, se = cost_fn.malis_weight_bdm_2D(bdm, lbl)

elapsed = time.time() - start
print "elapsed time is {} sec".format(elapsed)


#%% plot the results
print "plot the images"
import matplotlib.pylab as plt
if is_constrained:
    mbdm = np.copy(bdm)
    mbdm[lbl>0] = 1
    plt.subplot(231)
    plt.imshow(1-mbdm, cmap='gray')
    plt.xlabel('merger constrained boundary map')
    
    sbdm = np.copy(bdm)
    sbdm[lbl==0] = 0
    plt.subplot(231)
    plt.imshow(1-sbdm, cmap='gray')
    plt.xlabel('splitter constrained boundary map')
else:
    plt.subplot(232)
    plt.imshow(1-bdm, cmap='gray')
    plt.xlabel('boundary map')
    plt.subplot(234)
    plt.imshow(lbl==0, cmap='gray')
    plt.xlabel('manual labeling')

# rescale to 0-1
def rescale(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr

def combine2rgb(bdm, w):
    # make a combined colorful image
    cim = np.zeros(bdm.shape+(3,), dtype='uint8')
    # red channel
    cim[:,:,0] = (rescale(1-bdm))*255
    # green channel
    cim[:,:,1] = rescale( w )*255
    return cim

# combine merging error with boundary map
rgbm = combine2rgb(bdm, np.log(me+eps))
plt.subplot(232)
plt.imshow( rgbm )
plt.xlabel('combine boundary map(red) and ln(merge weight)(green)')

# combine merging error with boundary map
rgbs = combine2rgb(bdm, np.log(se+eps))
plt.subplot(235)
plt.imshow( rgbs )
plt.xlabel('combine boundary map(red) and ln(split weight)(green)')

# combine merging error with boundary map
rgbm = combine2rgb(bdm, me)
plt.subplot(233)
plt.imshow( rgbm )
plt.xlabel('combine boundary map(red) and merge weight(green disk)')
# plot disk to illustrate the weight strength
# rescale to 0-1
rme = rescale(me) * Dm
y,x = np.nonzero(rme)
r = rme[(y,x)]
plt.scatter(x,y,r, c='g', alpha=0.8)

# combine merging error with boundary map
rgbs = combine2rgb(bdm, se)
plt.subplot(236)
plt.imshow( rgbs )
plt.xlabel('combine boundary map(red) and split weight(green disk)')
# plot disk to illustrate the weight strength
# rescale to 0-1
rse = rescale(se) * Ds
y,x = np.nonzero(rse)
r = rse[(y,x)]
plt.scatter(x,y,r, c='g', alpha=0.8)

plt.show()
