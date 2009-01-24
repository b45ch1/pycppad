#!/usr/bin/env python
# ---------------------------------------------------------------------
# User options
# Can use ../cppad-20081128 for cppad_include_dir, but it generates a warning, 
# while the current version of CppAD dose not.
cppad_include_dir = '/home/brad/CppAD/trunk' # contains cppad/cppad.hpp
boost_lib_dir     = '/usr/lib'               # contains Boost::python library
boost_python_lib  = 'boost_python'           # name of Boost::python library
pycppad_debug     = True                     # compile for debugging True/False
# ---------------------------------------------------------------------
# This file follows the specifications in
# http://docs.python.org/distutils/setupscript.html
from distutils.core import setup, Extension
#
brad_email             = 'bradbell @ seanet dot com'
sebastian_email        = 'sebastian dot walter @ gmail dot com'
package_name           = 'pycppad'
package_version        = '20090000'
package_author         = 'Sebastian F. Walter and Bradley M. Bell'
package_author_email   = brad_email + ' , ' + sebastian_email 
package_url            = 'http://github.com/b45ch1/pycppad/tree/master'
package_description    = 'python Algorihtmic Differentiation using CppAD'
#
python_modules         = [ 'cppad' ]
#
extension_name         = 'cppad_'
extension_include_dirs = [ cppad_include_dir ]
extension_library_dirs = [ boost_lib_dir ]
extension_libraries    = [ boost_python_lib ]
if pycppad_debug : extension_undef_macros = [ 'NDEBUG' ]
else             : extension_undef_macros = [ ]
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
  name         = package_name               ,
  version      = package_version            ,
  author       = package_author             ,
  author_email = package_author_email       ,
  url          = package_url                ,
  py_modules   = python_modules             , 
  ext_modules  = extension_modules          ,
  packages     = [ package_name ]           ,
  package_dir  = { package_name : './' }
)
