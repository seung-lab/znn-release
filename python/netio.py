
import numpy as np
import h5py
import pyznn
import os.path, shutil
from utils import assert_arglist

np_array_fields = ("filters","biases","size","stride")
def save_opts(opts, filename):
    #Note: opts is a tuple of lists of dictionaries
    f = h5py.File(filename, 'w')

    for group_type in range(len(opts)): #nodes vs. edges

        #loop over group dict list
        for layer in opts[group_type]:

            #each layer is a dict
            layer_name = layer["name"]
            data_size = layer["size"]
            #Init

            #create a dataset for the filters/biases
            fields = layer.keys()
            if "filters" in fields:

                filters_dset_name = "/%s/%s" % (layer_name, "filters")
                f.create_dataset(filters_dset_name, data=layer["filters"][0])

                momentum_dset_name = "/%s/%s" % (layer_name, "momentum_vol")
                f.create_dataset(momentum_dset_name, data=layer["filters"][1])

            elif "biases" in fields:

                biases_dset_name = "/%s/%s" % (layer_name, "biases")
                f.create_dataset(biases_dset_name, data=layer["biases"][0])

                momentum_dset_name = "/%s/%s" % (layer_name, "momentum_vol")
                f.create_dataset(momentum_dset_name, data=layer["biases"][1])

            if "size" in fields:

                dset_name = "/%s/%s" % (layer_name, "size")
                data = np.array(layer["size"])

                f.create_dataset(dset_name, data=data)

            if "stride" in fields:

                dset_name = "/%s/%s" % (layer_name, "stride")
                data = np.array(layer["stride"])

                f.create_dataset(dset_name, data=data)


            for field in layer:

                if field in np_array_fields:
                    continue #already taken care of

                attr_name = "/%s/%s" % (layer_name, field)
                f[attr_name] = layer[field]

            #Final flag for node_group type
            group_type_name = "/%s/%s" % (layer_name, "group_type")
            f[group_type_name] = ("node","edge")[group_type]

    f.close()

def save_network(network, filename, num_iters=None):
    '''Saves a network under an h5 file. Appends the number
    of iterations if passed, and updates a "current" file with
    the most recent (uncorrupted) information'''

    # get directory name from file name
    archive_directory_name = os.path.dirname( filename )
#    filename = os.path.basename( filename )
    if not os.path.exists(archive_directory_name):
        os.mkdir(archive_directory_name)

#    filename = "{}/{}".format(archive_directory_name, filename)

    root, ext = os.path.splitext(filename)

    filename_current = "{}{}{}".format(root, '_current', ext)

    if num_iters is not None:
        filename = "{}{}{}{}".format(root, '_', num_iters, ext)

    save_opts(network.get_opts(), filename)

    # Overwriting most current file with completely saved version
    shutil.copyfile(filename, filename_current)

def load_opts(filename):
    '''Loads a pyopt structure (tuple of list of dicts) from a stored h5 file'''

    f = h5py.File(filename, 'r')

    node_opts = []
    edge_opts = []

    #each file has a collection of h5 groups which details a
    # network layer
    for group in f:

        layer = {}

        #each network layer has a number of fields
        for field in f[group]:

            #h5 file loads unicode strings, which causes issues later
            # when passing to c++
            field = str(field)

            dset_name = "/%s/%s" % (group, field)

            if field == "filters":

                momentum_dset_name = "/%s/%s" % (group, "momentum_vol")

                layer["filters"] = (
                    f[dset_name].value,
                    f[momentum_dset_name].value
                    )

            elif field == "biases":

                momentum_dset_name = "/%s/%s" % (group, "momentum_vol")

                layer["biases"] = (
                    f[dset_name].value,
                    f[momentum_dset_name].value
                    )

            elif field == "size":

                layer["size"] = tuple(f[dset_name].value)

            elif field == "stride":

                layer["stride"] = tuple(f[dset_name].value)

            elif field == "group_type":

                #group_type is handled after the dict is complete
                # (after the if statements here)
                continue

            elif field == "momentum_vol":

                #This should be loaded by the filters or biases option
                continue

            else:

                layer[field] = f[dset_name].value

        #Figuring out where this layer belongs (group_type)
        group_type_name = "/%s/%s" % (group, "group_type")
        if f[group_type_name].value == "node":
            node_opts.append(layer)
        else:
            edge_opts.append(layer)

    return (node_opts, edge_opts)

def load_network( params=None, train=True, hdf5_filename=None,
    network_specfile=None, output_patch_shape=None, num_threads=None,
    optimize=None ):
    '''
    Loads a network from an hdf5 file.

    The function will define the loading process by a parameter object
    (as generated by the front_end.parse function), or by the specified options.

    If both a parameter object and any optional arguments are specified,
    the parameter object will form the default options, and those will be
    overwritten by the other optional arguments
    '''
    #Need to specify either a params object, or all of the other optional args
    params_defined = params is not None

    #"ALL" optional args excludes train
    all_optional_args_defined = all([
        arg is not None for arg in
        ( hdf5_filename, network_specfile, output_patch_shape,
            num_threads, optimize )
        ])

    assert ( params_defined or all_optional_args_defined )

    #Defining phase argument by train argument
    phase = int(not train)

    #If a params object exists, then those options are the default
    if params is not None:

        if train:
            _hdf5_filename = params['train_load_net']
            _output_patch_shape = params['train_outsz']
            _optimize = params['is_train_optimize']
        else:
            _hdf5_filename = params['forward_net']
            _output_patch_shape = params['forward_outsz']
            _optimize = params['is_forward_optimize']

        _network_specfile = params['fnet_spec']
        _num_threads = params['num_threads']

    #Overwriting defaults with any other optional args
    if hdf5_filename is not None:
        _hdf5_filename = hdf5_filename
    if network_specfile is not None:
        _network_specfile = network_specfile
    if output_patch_shape is not None:
        _output_patch_shape = output_patch_shape
    if num_threads is not None:
        _num_threads = num_threads

    #ACTUAL LOADING FUNCTIONALITY
    #If the file doesn't exist, init a new network
    if os.path.isfile( _hdf5_filename ):

        options = load_opts(_hdf5_filename)
        return pyznn.CNet(options, _network_specfile, _output_patch_shape,
                _num_threads, _optimize, phase)

    else:
        return init_network( params, train, _network_specfile, _output_patch_shape,
                _num_threads, _optimize )

def init_network( params=None, train=True, network_specfile=None,
            output_patch_shape=None, num_threads=None, optimize=None ):
    '''
    Initializes a random network using the Boost Python interface and configuration
    file options.

    The function will define this network by a parameter object
    (as generated by the front_end.parse function), or by the specified options.

    If both a parameter object and any optional arguments are specified,
    the parameter object will form the default options, and those will be
    overwritten by the other optional arguments
    '''
    #Need to specify either a params object, or all of the other optional args
    #"ALL" optional args excludes train
    assert_arglist(params, 
                [network_specfile, output_patch_shape, 
                num_threads, optimize]
                )

    #Defining phase argument by train argument
    phase = int(not train)

    #If a params object exists, then those options are the default
    if params is not None:

        if train:
            _output_patch_shape = params['train_outsz']
            _optimize = params['is_train_optimize']
        else:
            _output_patch_shape = params['forward_outsz']
            _optimize = params['is_forward_optimize']

        _network_specfile = params['fnet_spec']
        _num_threads = params['num_threads']

    #Overwriting defaults with any other optional args
    if network_specfile is not None:
        _network_specfile = network_specfile
    if output_patch_shape is not None:
        _output_patch_shape = output_patch_shape
    if num_threads is not None:
        _num_threads = num_threads
    if optimize is not None:
        _optimize = optimize

    return pyznn.CNet(_network_specfile, _output_patch_shape,
                    _num_threads, _optimize, phase)
