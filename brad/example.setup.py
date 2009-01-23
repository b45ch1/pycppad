#!/usr/bin/env python
# ---------------------------------------------------------------------
# User options
# Can use ../cppad-20081128 for cppad_include_dir, but it generates a warning, 
# while the current version of CppAD dose not.
cppad_include_dir = '/home/brad/CppAD/trunk'
boost_lib_dir     = '/usr/lib'
boost_python_lib  = 'boost_python'
# ---------------------------------------------------------------------
# This file follows the specifications in
# http://docs.python.org/distutils/setupscript.html
from distutils.core import setup, Extension
#
package_version        = '20090000'
package_author         = 'Sebastian F. Walter and Bradley M. Bell'
package_url            = 'http://github.com/b45ch1/pycppad/tree/master'
package_description    = 'python Algorihtmic Differentiation using CppAD'
#
extension_name         = 'cppad_'
extension_include_dirs = [ cppad_include_dir ]
extension_library_dirs = [ boost_lib_dir ]
extension_libraries    = [ boost_python_lib ]
extension_undef_macros = [ 'NDEBUG' ]
extension_sources      = [ 
  'adfun.cpp'       ,
  'pycppad.cpp'     ,
  'vec2array.cpp'   ,
  'vector.cpp' 
]
extension_modules = [ Extension( 
  extension_name                        , 
  extension_sources                     ,
  include_dirs = extension_include_dirs ,
  library_dirs = extension_library_dirs ,
  libraries    = extension_libraries    ,
  undef_macros = extension_undef_macros
) ]
#
setup(
  name        = extension_name    ,
  version     = package_version   ,
  author      = package_author    ,
  url         = package_url       ,
  ext_modules = extension_modules
)
