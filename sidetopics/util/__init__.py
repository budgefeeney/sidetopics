import os
import numpy as np
import pyximport

from os.path import dirname, realpath
from distutils.core import setup


# os.environ['CC'] = os.environ['HOME'] + '/bin/cc'
pyximport.install( \
    build_in_temp=False, \
    inplace=True, \
    build_dir=dirname(dirname(realpath(__file__))) + '/lib', \
    setup_args={ \
        'include_dirs':       np.get_include(), \
        'libraries':          [('m', dict()), ('gomp', dict()), ('gsl', dict()), ('gslcblas', dict())], \
        'extra_compile_args': '-fopenmp'
    }, \
    reload_support=True)