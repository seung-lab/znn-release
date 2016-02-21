#!/usr/bin/env python
__doc__ = """

AWS S3 Data Handling Functions

Nicholas Turner <nturner@cs.princeton.edu>, 2016
"""

import numpy as np
import os, subprocess



def get_sample_section_names(config, net, sample_num):
    '''
    Fetches the sections of the configuration file which
    contain data pathnames for a specific sample

    Uses the network to find the sample options
    '''

    sample_name = "sample%d" % sample_num

    #Retrieves the network inputs and outputs, and then
    # extracts the keys to derive the names of the config
    # sample fields 
    image_names = net.get_inputs_setsz().keys()
    label_names = net.get_outputs_setsz().keys()

    #For each name above, find the section name within the config file
    image_section_names = ["image%d" % config.getint(sample_name, name) 
                                            for name in image_names]
    
    #Some datasets don't have labels, so we need to be a bit more careful
    # with those
    label_section_names = []
    for name in label_names:
        if config.has_option(sample_name, name):
            label_section_names.append( "label%d" % config.getint(sample_name, name) )

    return image_section_names + label_section_names

def get_section_filenames(config, section_name):
    '''
    Inspects a section of the configuration file, and 
    returns its constituent filenames
    '''
    #configuration file convention is to delimit filenames
    # by a '\n'
    fnames = config.get( section_name, 'fnames').split('\n')

    #if the section contains mask filenames, add them to the filename list
    if config.has_option( section_name, 'fmasks' ):
        fnames.extend( config.get( section_name, 'fmasks' ).split('\n') )

    return fnames

def get_sample_filenames(config, net, sample_num):
    '''
    Inspects a sample of the configuration file, 
    and returns its constituent filenames
    '''
    section_names = get_sample_section_names(config, net, sample_num)
    
    filenames = []
    for section_name in section_names:
        filenames.extend( get_section_filenames(config, section_name) )

    return filenames

def get_s3_section_pathnames(config, section_name):
    '''
    Inspects a section of the configuration file, and
    returns its constituent filenames, paired along with
    a file's s3 pathname
    '''

    fnames = config.get( section_name, 'fnames').split('\n')
    s3_fnames = config.get( section_name, 's3_fnames').split('\n')
    
    if config.has_option( section_name, 'fmasks'):
        fnames.extend( config.get( section_name, 'fmasks' ).split('\n') )
        s3_fnames.extend( config.get( section_name, 's3_fmasks' ).split('\n') )

    assert len(fnames) == len(s3_fnames)

    return zip(s3_fnames, fnames)

def section_on_disk(config, section_name):
    '''
    Inspects all files within a section of the configuration file,
    and returns whether or not they are currently on disk
    '''
    fnames = get_section_filenames(config, section_name)

    return all( [os.path.isfile(fname) for fname in fnames] )

def sample_on_disk(config, net, sample_num):
    '''
    Inspects all files within a sample field of the configuration file,
    and returns whether or not they are currently available on disk
    '''
    section_names = get_sample_section_names(config, net, sample_num)

    return all( [section_on_disk(config, name)
                         for name in section_names] )

def copy_file_from_S3(s3_pathname, local_pathname):
    '''
    Actually copies things if the file doesn't already exist locally
    '''

    s3_cp_command = ['aws','s3','cp',s3_pathname,local_pathname]

    if not os.path.isfile(local_pathname):
        print "copying file..."

        #creating the directory if it doesn't exist already
        dirname = os.path.dirname(local_pathname)
        try:
            print "making directory"
            os.makedirs( dirname )
        except:
            pass

        subprocess.call(s3_cp_command)
        print "should be done"
    else:
        print "local_pathname already exists!"

def copy_section_from_S3(config, section_name):
    '''
    Copies the fnames within a section to the local disk
    '''

    s3_fname_pairs = get_s3_section_pathnames(config, section_name)

    for (s3_pathname, local_pathname) in s3_fname_pairs:
        copy_file_from_S3(s3_pathname, local_pathname)

def copy_sample_from_S3(config, net, sample_num):
    '''
    Copies all files from a sample onto local directories as specified
    by the s3_fnames and fnames fields
    '''

    section_names = get_sample_section_names(config, net, sample_num)

    for section in section_names:
        copy_section_from_S3(config, section)

def rm_sample(config, net, sample_num, active_sample_num):
    '''
    Removes the files for a sample from the disk
    '''
    sample_filenames = get_sample_filenames(config, net, sample_num)
    active_sample_filenames = get_sample_filenames(config, net, active_sample_num)

    for filename in sample_filenames:
        if filename not in active_sample_filenames:
            if os.path.isfile(filename):
                print "removing %s" % filename
                os.remove(filename)
        else:
            print "%s currently active!" % filename

