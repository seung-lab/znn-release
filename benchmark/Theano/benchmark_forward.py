import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import theano.tensor.nnet.conv3d2d
from theano.sandbox.cuda import fftconv
from theano.tensor.nnet.Conv3D import conv3D
from maxpool3d import max_pool_3d
from theano.tensor.shared_randomstreams import RandomStreams


class Conv3DLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, pool_shape):

        self.input = input

        # initialize weights with random weights
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-.1, high=.1, size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # filter stride
        d = theano.shared(numpy.ndarray(shape=(3,),dtype=int))
        d.set_value([1, 1, 1])

        # convolve input feature maps with filters
        # # theano.tensor.nnet.conv3D
        # conv_out = conv3D(
        #     V=self.input,
        #     W=self.W,
        #     b=self.b,
        #     d=d)
        # theano.tensor.nnet.conv3d2d.conv3d
        conv_out = T.nnet.conv3d2d.conv3d(
            signals=self.input,
            filters=self.W,
            signals_shape=image_shape,
            filters_shape=filter_shape,
            border_mode='valid')

        activation = T.tanh(conv_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x'))

        if pool_shape == (1,1,1):
            self.output = activation
        else:
            self.output = max_pool_3d(activation, pool_shape)

        # store parameters of this layer
        self.params = [self.W, self.b]


def random_matrix(shape, np_rng, name=None):
    return theano.shared(numpy.require(np_rng.randn(*shape), dtype=theano.config.floatX),
            borrow=True, name=name)


def evaluate_convnet(learning_rate=0.01, n_epochs=100, nlayer=5, width=[10], fxy=[5], fz=[5], pxy=[1], pz=[1], outxy=1, outz=1):

    # random number generator for weight initialization
    rng = numpy.random.RandomState(23455)

    # preprocessing
    if len(width) < nlayer:
        width = [width[0]] * nlayer
    if len(fxy) < nlayer:
        fxy = [fxy[0]] * nlayer
    if len(fz) < nlayer:
        fz = [fz[0]] * nlayer
    if len(pxy) < nlayer:
        pxy = [pxy[0]] * nlayer
    if len(pz) < nlayer:
        pz = [pz[0]] * nlayer

    # compute input size
    fmapxy = [outxy]
    fmapz = [outz]
    for i in xrange(nlayer-1,-1,-1):
        fmapxy.append((pxy[i]*fmapxy[-1]) + (fxy[i] - 1))
        fmapz.append((pz[i]*fmapz[-1]) + (fz[i] - 1))
    fmapxy = fmapxy[::-1]
    fmapz = fmapz[::-1]

    print '\n input size = {} x {} x {}'.format(fmapxy[0], fmapxy[0], fmapz[0])
    print 'output size = {} x {} x {}'.format(outxy, outxy, outz)

    # # random input
    # data_x = random_matrix((n_epochs, fmapz[0], 1, fmapxy[0], fmapxy[0]), rng, 'data_x')
    # data_y = random_matrix((n_epochs, outz, width[-1], outxy, outxy), rng, 'data_y')
    # idx = T.lscalar()
    # x = data_x[idx]
    # y = data_y[idx]

    # derterministic input
    x = random_matrix((1, fmapz[0], 1, fmapxy[0], fmapxy[0]), rng, 'x')
    y = random_matrix((1, outz, width[-1], outxy, outxy), rng, 'y')

    # patch size
    patch_sz = outxy*outxy*outz

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '\n... building the model'

    # initial hidden layer
    layer0 = Conv3DLayer(
        rng,
        input=x,
        image_shape=(1, fmapz[0], 1, fmapxy[0], fmapxy[0]),
        filter_shape=(width[0], fz[0], 1, fxy[0], fxy[0]),
        pool_shape=(pz[0],pxy[0],pxy[0]))
    layers = [layer0]
    params = layer0.params

    # remaining hidden layers
    for i in xrange(1,nlayer):
        layers.append(Conv3DLayer(
            rng,
            input=layers[-1].output,
            image_shape=(1, fmapz[i], width[i-1], fmapxy[i], fmapxy[i]),
            filter_shape=(width[i], fz[i], width[i-1], fxy[i], fxy[i]),
            pool_shape=(pz[i],pxy[i],pxy[i])))
        params = params + layers[-1].params

    # # random input
    # train_model = theano.function([idx], layers[-1].output)

    # deterministic input
    train_model = theano.function([], layers[-1].output)

    # theano.printing.debugprint(train_model, print_type=True)

    ###############
    # TRAIN MODEL #
    ###############
    print '... forward pass'
    epoch = 0
    done_looping = False
    time_accum = 0
    while (epoch < n_epochs) and (not done_looping):
        start_time = timeit.default_timer()
        #prop = train_model(epoch)# random input
	prop = train_model() # deterministic iput
        epoch = epoch + 1
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        print 'epoch {} took {}'.format(epoch, elapsed)
        time_accum = time_accum + elapsed

    print 'Forward pass complete.'
    print 'Speed: {}'.format(time_accum/n_epochs)

if __name__ == '__main__':

    # m96
    #nlayer=6
    #width=[80, 80, 80, 80, 80, 3]
    #fxy=[8, 9, 9, 9, 9, 9]
    #fz =[8, 9, 9, 9, 9, 9]
    #pxy=[2, 2, 1, 1, 1, 1]
    #pz =[2, 2, 1, 1, 1, 1]

    # m76
    nlayer=6
    width=[80,80,80,80,80, 3]
    fxy  =[ 6, 7, 7, 7, 7, 7]
    fz   =[ 6, 7, 7, 7, 7, 7]
    pxy  =[ 2, 2, 1, 1, 1, 1]
    pz   =[ 2, 2, 1, 1, 1, 1]

    # m56
    #nlayer=7
    #width=[80,80,80,80,80,80, 3]
    #fxy  =[ 4, 5, 5, 5, 5, 5, 5]
    #fz   =[ 4, 5, 5, 5, 5, 5, 5]
    #pxy  =[ 2, 2, 2, 1, 1, 1, 1]
    #pz   =[ 2, 2, 2, 1, 1, 1, 1]

    # m36
    #nlayer=7
    #width=[80,80,80,80,80,80, 3]
    #fxy  =[ 2, 3, 3, 3, 3, 3, 3]
    #fz   =[ 2, 3, 3, 3, 3, 3, 3]
    #pxy  =[ 2, 2, 2, 1, 1, 1, 1]
    #pz   =[ 2, 2, 2, 1, 1, 1, 1]

    # output patch size
    outxy=int(sys.argv[1])
    outz=int(sys.argv[2])

    evaluate_convnet(nlayer=nlayer,width=width,fxy=fxy,fz=fz,pxy=pxy,pz=pz,outxy=outxy,outz=outz)
