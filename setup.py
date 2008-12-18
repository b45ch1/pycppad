#!/usr/bin/env python
# build with: $ python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs
from numpy.distutils.misc_util import Configuration


include_dirs = [get_numpy_include_dirs(),'./cppad-20081128']
extra_compile_args = ['-ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB']
library_dirs = ['/data/walter/opt_software/boost_1_34_1/bin.v2/libs/python/build/gcc-4.2.1/release']
libraries = ['boost_python-gcc42-1_34_1']

config = Configuration('',parent_package='',top_path='')
config.add_extension(
	name = '_cppad',
	sources = ['py_cppad.cpp', 'num_util.cpp'],
	include_dirs = include_dirs,
	extra_compile_args = extra_compile_args,
	library_dirs = library_dirs,
	runtime_library_dirs = library_dirs,
	libraries = libraries
)
#include_dirs, define_macros, undef_macros, library_dirs, libraries, runtime_library_dirs, extra_objects, extra_compile_args, extra_link_args, export_symbols, swig_opts, depends, language, f2py_options, module_dirs, extra_info.

setup(**config.todict())
