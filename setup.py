#!/usr/bin/env python
import os
# ---------------------------------------------------------------------
# The code below is included verbatim in omh/install.omh
# BEGIN USER SETTINGS
# Directory where Boost Python library and include files are located
boost_python_lib_dir = '/usr/lib' 
boost_python_include_dir= '/usr/include'
# Name of the Boost Python library in boost_python_lib_dir.
# Must begin with 'lib' and end with '.so' (under unix) or '.a' (under cygwin).
boost_python_lib_name_cygwin  = 'libboost_python-mt.a' 
boost_python_lib_name_darwin  = 'libboost_python.dylib'
boost_python_lib_name_unix    = 'libboost_python.so'
# The CppAD tarball will be downloaded into this directory
# and it will be extraced to a subdirectory called  cppad-*
cppad_parent_dir = 'external'
# 
# END USER SETTINGS
# ---------------------------------------------------------------------
# See http://docs.python.org/distutils/setupscript.html
# for documentation on how to write the script setup.py
#
# See  http://docs.python.org/install/index.html
# for documentation on how to use the script setup.py 
# ---------------------------------------------------------------------
# Values in setup.py that are replaced by build.sh
package_version    = '20111016'
cppad_tarball      = 'cppad-20110101.5.gpl.tgz'
cppad_download_dir = 'http://www.coin-or.org/download/source/CppAD'
# ---------------------------------------------------------------------
import re
import sys
import platform
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
# ---------------------------------------------------------------------
# determine if this is a cygwin system
uname     = (( platform.uname() )[0] )[0:6]
is_cygwin = (uname == 'CYGWIN')
is_darwin = (uname == 'Darwin')
# ---------------------------------------------------------------------
# determine the install directory and the include directory
match=re.search('([^-]*-[0-9]*[.][0-9]*)[.].*', cppad_tarball)
cppad_dir         = match.group(1)
cppad_include_dir = cppad_parent_dir + '/' + cppad_dir
# ---------------------------------------------------------------------
# check for the specified boost-python library
if is_cygwin :
  boost_python_lib_name = boost_python_lib_name_cygwin
  m = re.search('lib([a-zA-Z0-9_-]+)[.]a', boost_python_lib_name)
  if m == None :
    print "boost_python_lib_name_cygwin must begin with 'lib' & end with '.a'"
elif is_darwin :
  boost_python_lib_name = boost_python_lib_name_darwin
  m = re.search('lib([a-zA-Z0-9_-]+)[.]dylib', boost_python_lib_name)
  if m == None :
    print "boost_python_lib_name_darwin must begin with 'lib' & end with '.dylib'"
else :
  boost_python_lib_name = boost_python_lib_name_unix
  m = re.search('lib([a-zA-Z0-9_-]+)[.]so', boost_python_lib_name)
  if m == None :
    print "boost_python_lib_name_unix must begin with 'lib' & end with '.so'"
boost_python_lib = m.group(1)
boost_python_lib_path = boost_python_lib_dir + '/' + boost_python_lib_name
if not os.access ( boost_python_lib_path , os.R_OK ) :
  print 'Cannot find the Boost Python library: ' + boost_python_lib_path 
  print 'Use the web page http://www.boost.org/ for information about'
  print 'boost::python.  Make sure that boost_python_lib_dir and'
  print 'boost_python_lib_name are set correctly at the beginning of the file'
  print 'setup.py.'
  exit(1)
# ---------------------------------------------------------------------
# make sure we have a copy of the soruce for the specified cppad version
build_source_dist = False
if len( sys.argv ) == 2 :
  if sys.argv[1]  == 'sdist' :
    build_source_dist = True 
if not build_source_dist :
  if not os.access( cppad_parent_dir , os.R_OK ) :
    cmd = 'mkdir ' + cppad_parent_dir
    os.system( cmd )
  if not os.access( cppad_include_dir , os.R_OK ) :
    print 'os.getcwd() = ' + os.getcwd()
    print 'Please wait while the proper version of CppAD is downloaded'
    cmd = '( cd ' + cppad_parent_dir + ';'
    cmd = cmd + ' curl -O ' + cppad_download_dir + '/' + cppad_tarball + ')'
    print cmd
    os.system( cmd )
    cmd = '( cd ' + cppad_parent_dir + ';'
    cmd = cmd + ' tar -xzf ' + cppad_tarball + ')'
    print cmd
    os.system( cmd )
  if not os.access( cppad_include_dir , os.R_OK ) :
    print 'Did not successfully retrieve the proper verison of CppAD'
    exit(1)
# ---------------------------------------------------------------------
try:
    doc_files    = os.listdir('doc')
    for i in range( len(doc_files) ):
        doc_files[i] = 'doc/' + doc_files[i]
    
except:
    doc_files = []
#
brad_email             = 'bradbell @ seanet dot com'
sebastian_email        = 'sebastian dot walter @ gmail dot com'
package_author         = 'Bradley M. Bell and Sebastian F. Walter'
package_author_email   = sebastian_email + ' , ' + brad_email
package_url            = 'http://github.com/b45ch1/pycppad/tree/master'
package_description    = 'Python Algorihtmic Differentiation Using CppAD'
package_data_files     = [ ('share/doc/pycppad', doc_files ) ]
package_license        = 'BSD'
#
cppad_extension_name         = 'pycppad' + '/cppad_'
cppad_extension_include_dirs = get_numpy_include_dirs()
cppad_extension_include_dirs.append( cppad_include_dir )
cppad_extension_include_dirs.append( boost_python_include_dir )
cppad_extension_library_dirs = [ boost_python_lib_dir ]
cppad_extension_libraries    = [ boost_python_lib ]
cppad_extension_sources      = [ 
  'pycppad' + '/adfun.cpp'       ,
  'pycppad' + '/pycppad.cpp'     ,
  'pycppad' + '/vec2array.cpp'   ,
  'pycppad' + '/vector.cpp' 
]
extension_modules = [ Extension( 
  cppad_extension_name                        , 
  cppad_extension_sources                     ,
  include_dirs = cppad_extension_include_dirs ,
  library_dirs = cppad_extension_library_dirs ,
  libraries    = cppad_extension_libraries    ,
) ]
#
setup(
  name         = 'pycppad'               ,
  version      = package_version            ,
  license      = package_license            ,
  description  = package_description        ,
  author       = package_author             ,
  author_email = package_author_email       ,
  url          = package_url                ,
  ext_modules  = extension_modules          ,
  packages     = [ 'pycppad' , 'pycppad' ]  ,
  package_dir  = { 'pycppad' : 'pycppad' }        ,
  data_files   = package_data_files
)
