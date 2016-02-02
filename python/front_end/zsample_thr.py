#!/usr/bin/env python
__doc__ = """

Threaded Versions of 
Dataset Class Interface (CSamples)

Nicholas Turner <nturner@cs.princeton.edu>, 2015
"""

import random
import utils
import s3_utils
import zsample
from threading import Thread

class LoaderThread(Thread):
    '''
    Thread class used within CThreadedSamples below

    Prepares a sample from disk using one of the CSample
    classes above (either CBoundarySample or CAffinitySample),
    and stores it within an attribute self.sample
    '''

    def __init__(self, config, pars, sample_num, net, outsz, log, class_obj):
        '''Mostly just stores info for later use'''
        Thread.__init__(self)

        #storing stuff
        self.config = config
        self.pars = pars
        self.sample_num = sample_num
        self.net = net
        self.outsz = outsz
        self.log = log
        self.class_obj = class_obj

        self.sample = None

    def run(self):
        '''Actually doing work'''

        self.sample = self.class_obj(
            self.config,
            self.pars,
            self.sample_num,
            self.net,
            self.outsz,
            self.log)


class CThreadedSamples(object):
    '''
    Samples Class which allows for a group of datasets which is altogether too
    large to fit in memory

    Features the same get_random_sample interface as CSample[s], but only features
    a single active dataset at once. The active dataset can be swapped with another
    random set through the swap_samples function. This in turn triggers another
    backup to be prepped from disk.
    '''

    def __init__(self, config, pars, ids, net, outsz, log=None):
        '''
        Stores parameters, and starts the first 
        sample loading process in the foreground
        '''

        #Storing inputs for use within loading functions
        self.config = config
        self.pars = pars
        #conversion to tuple allows for indexing
        # within random.choice when we select a random sample
        self.ids = tuple(ids) 
        self.net = net
        self.outsz = outsz
        self.log = log

        #Selecting which output sample class to load
        self.sample_class = None
        if 'bound' in pars['out_type']:
            self.sample_class = zsample.CBoundarySample
        elif 'aff' in pars['out_type']:
            self.sample_class = zsample.CAffinitySample
        else:
            raise NameError('invalid output type')

        #Loading initial dataset
        self.active_sample = self.load_sample_direct()

        #Starting loading of backup dataset
        self.backup_thread = self.load_sample_thread()

    def choose_random_sample_num(self):
        '''Returns a random sample id'''
        return random.choice( self.ids )

    def load_sample_direct(self):
        '''
        Prepares a CSample object from disk in the foreground.
        Returns the CSample object itself
        '''
        sample_num = self.choose_random_sample_num()

        #Initialize an instance of the stored sample class
        # using the stored info
        return self.sample_class(
                        self.config,
                        self.pars,
                        sample_num,
                        self.net,
                        self.outsz,
                        self.log)

    def load_sample_thread(self):
        '''
        Prepares a CSample object from disk within a LoaderThread.
        Returns a thread which will (eventually) contain the CSample
        object

        self, config, pars, sample_num, net, outsz, log, class_obj)
        '''
        sample_num = self.choose_random_sample_num()

        #Initializing the thread
        thread = LoaderThread(
            self.config,
            self.pars,
            sample_num,
            self.net,
            self.outsz,
            self.log,
            self.sample_class
            )

        #Starting the read from disk
        thread.start()

        #Returning the pointer
        return thread

    def backup_ready(self):
        '''Returns whether the backup is ready to be swapped in'''
        #If the backup thread is still running, it has more work to do,
        # and the sample object should be None
        return not self.backup_thread.is_alive()

    def swap_samples(self):
        '''
        Swaps the backup sample to be active if it's ready,
        and otherwise does nothing
        '''
        if self.backup_ready():
            print "Swapping active dataset"
            self.active_sample = self.backup_thread.sample
            #Starting the next load
            self.backup_thread = self.load_sample_thread()
        else:
            print "Swap not ready!"

    def get_active_sample_id(self):
        '''Returns the id of the active sample'''
        return self.active_sample.sec_name

    def get_random_sample(self):
        '''
        Fetches a random input and output patch from the active sample
        '''
        return self.active_sample.get_random_sample()


#===================================================
#Samples adapted to use AWS S3

class S3Copy_LoaderThread(Thread):

    def __init__(self, config, pars, sample_num, net, outsz, log, class_obj):
        '''Same as above'''
        Thread.__init__(self)

        #storing stuff
        self.config = config
        self.pars = pars
        self.sample_num = sample_num
        self.net = net
        self.outsz = outsz
        self.log = log
        self.class_obj = class_obj

        self.sample = None

    def run(self):
        '''
        Checks to see if sample is on disk
        copy it from s3 if it's not, and then
        start the loader thread
        '''
        s3_utils.copy_sample_from_S3(
            self.config,
            self.net,
            self.sample_num)

        #Initializing the loader thread
        thread = LoaderThread(
            self.config,
            self.pars,
            self.sample_num,
            self.net,
            self.outsz,
            self.log,
            self.class_obj
            )

        #Running it in the current process
        thread.run()
        self.sample = thread.sample 


class CThreadedSamples_S3(CThreadedSamples):

    def __init__(self, config, pars, ids, net, outsz, log=None):
        '''
        Nothing new
        '''
        CThreadedSamples.__init__(self, config, pars,
                                ids, net, outsz, log)

    def load_sample_direct(self):
        '''
        Prepares a CSample object from disk directly.
        Returns the resulting CSample object
        '''
        #it helps to store the sample_number for later here
        # for when we remove the files
        self.active_sample_num = self.choose_random_sample_num()

        s3_utils.copy_sample_from_S3(self.config, self.net, self.active_sample_num)

        return self.sample_class(
                        self.config,
                        self.pars,
                        self.active_sample_num,
                        self.net,
                        self.outsz,
                        self.log)

    def load_sample_thread(self):
        '''
        Prepares a CSample object from disk within a LoaderThread.
        Returns a thread which will (eventually) contain the CSample
        object

        self, config, pars, sample_num, net, outsz, log, class_obj)
        '''
        #it helps to store the sample_number for later here
        # for when we remove the files
        self.backup_sample_num = self.choose_random_sample_num()

        #Initializing the thread
        thread = S3Copy_LoaderThread(
            self.config,
            self.pars,
            self.backup_sample_num,
            self.net,
            self.outsz,
            self.log,
            self.sample_class
            )

        #Starting the read from disk
        thread.start()

        #Returning the pointer
        return thread

    def swap_samples(self):
        '''
        Only change from above is to remove the active dataset before the swap
        '''
        if self.backup_ready():
            print "Swapping active dataset"
            self.active_sample = self.backup_thread.sample

            #Removing the non-overlapping files from the old
            # sample
            s3_utils.rm_sample(self.config, self.net, 
                self.active_sample_num, self.backup_sample_num)

            self.active_sample_num = self.backup_sample_num

            #Starting the next load
            self.backup_thread = self.load_sample_thread()
        else:
            print "Swap not ready!"

        
