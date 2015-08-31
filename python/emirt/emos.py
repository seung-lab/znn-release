# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 16:27:15 2015

@author: jingpeng
"""

def mkdir_p(path):
    """ 'mkdir -p' in Python """
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise