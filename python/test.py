import numpy as np
import pyznn
import emirt
import time
# parameters
ftrn = "../dataset/ISBI2012/data/original/train-volume.tif"
flbl = "../dataset/ISBI2012/data/original/train-labels.tif"
fnet_spec = '../networks/srini2d.znn'
# learning rate
eta = 0.01
# output size
outsz = np.asarray([1,5,5])
# number of threads
num_threads = 7

# prepare input
vol = emirt.io.imread(ftrn).astype('float32')
lbl = emirt.io.imread(flbl).astype('float32')
# normalize the training volume
vol = emirt.volume_util.norm( vol )
lbl = (lbl>0.5).astype('float32')

print "output volume size: {}x{}x{}".format(outsz[0], outsz[1], outsz[2])
net = pyznn.CNet(fnet_spec, outsz[0],outsz[1],outsz[2],num_threads)
net.set_eta( eta / float(outsz[0] * outsz[1] * outsz[2]) )

# compute inputsize and get input
fov = np.asarray(net.get_fov())
print "field of view: {}x{}x{}".format(fov[0],fov[1], fov[2])
insz = fov + outsz - 1

def get_sample( vol, insz, lbl, outsz ):
    half_in_sz  = insz.astype('uint32')  / 2
    half_out_sz = outsz.astype('uint32') / 2
    # margin consideration for even-sized input
    margin_sz = half_in_sz - (insz%2)
    set_sz = vol.shape - margin_sz - half_in_sz
    # get random location
    loc = np.zeros(3)
    loc[0] = np.random.randint(half_in_sz[0], half_in_sz[0] + set_sz[0]-1)
    loc[1] = np.random.randint(half_in_sz[1], half_in_sz[1] + set_sz[1]-1)
    loc[2] = np.random.randint(half_in_sz[2], half_in_sz[2] + set_sz[2]-1)
    # extract volume
    vol_in  = vol[  loc[0]-half_in_sz[0]  : loc[0]-half_in_sz[0] + insz[0],\
                    loc[1]-half_in_sz[1]  : loc[1]-half_in_sz[1] + insz[1],\
                    loc[2]-half_in_sz[2]  : loc[2]-half_in_sz[2] + insz[2]]
    lbl_out = lbl[  loc[0]-half_out_sz[0] : loc[0]-half_out_sz[0]+outsz[0],\
                    loc[1]-half_out_sz[1] : loc[1]-half_out_sz[1]+outsz[1],\
                    loc[2]-half_out_sz[2] : loc[2]-half_out_sz[2]+outsz[2]]
    return (vol_in, lbl_out)


def square_loss(prop, lbl):
    """
    compute square loss 
    """
    cls = float(np.count_nonzero( (prop>0.5)!= lbl ))
    
    grdt = prop.astype('float32')
    grdt = grdt - lbl
    err = np.sum( grdt * grdt ).astype('float32')
    grdt = grdt * 2
    return (err, cls, grdt)


err = 0;
cls = 0;
# get gradient
for i in xrange(1,1000000):
    vol_in, lbl_out = get_sample( vol, insz, lbl, outsz )
    
    # forward pass
    prop = net.forward(vol_in)
    print "prop in python:"
    print prop
    time.sleep(1)
        
    cerr, ccls, grdt = square_loss( prop, lbl_out )  
    err = err + cerr
    cls = cls + ccls  
    # run backward pass
    net.backward(grdt)
    
    if i%1000==0:
        err = err / float(1000 * outsz[0] * outsz[1] * outsz[2]) 
        cls = cls / float(1000 * outsz[0] * outsz[1] * outsz[2])
        print "sqerr : {},    clserr: {}".format(err, cls)
        err = 0
        cls = 0
        
#%% visualization
com = emirt.show.CompareVol((vol_in, lbl_out))
com.vol_compare_slice()