#!/usr/bin/env python
__doc__ = """
compile script using cython
usage:
    python compile.py script

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import os
import sys
assert(len(sys.argv)==2)
script = os.path.basename( sys.argv[1] )

os.system("cython " + script + ".py")
os.system("gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o " \
            + script + ".so " + script + ".c")
