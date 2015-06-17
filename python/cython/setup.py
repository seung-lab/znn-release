from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# building command: python setup.py build_ext --inplace

setup(ext_modules = cythonize(Extension(
           "pyznn",                                # the extesion name
           # the Cython source and additional C++ source files
           sources=["pyznn.pyx", "options/options.hpp", "network/parallel/network.hpp"],
           language="c++",
           # generate and compile C++ code
           include_dirs=["../", "../src/include", "../zi", "/usr/people/jingpeng/libs/boost/include/"],
           libraries=["stdc++", "fftw3", "pthread", "rt", "fftw3_threads", "boost_program_options", "boost_regex", "boost_filesystem", "boost_system", "boost_timer"],
           extra_compile_args=['-g'],
           extra_link_args=['-g', "-L../../ -L/usr/people/jingpeng/libs/boost/lib64/"]
      )))
