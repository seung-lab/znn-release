
import numpy as np
import h5py
import pyznn
import os.path, shutil

archive_directory_name = 'ARCHIVE'

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

    if not os.path.exists(archive_directory_name):
        os.mkdir(archive_directory_name)

    filename = "{}/{}".format(archive_directory_name, filename)

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

def load_network(hdf5_filename, fnet_spec, outsz, num_threads):
    opts = load_opts(hdf5_filename)
    return pyznn.CNet(opts, fnet_spec, outsz, num_threads)
