#!/usr/bin/env python
import os
import sys
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
# ---------------------------------------------------------------------
# The code below is included verbatim in omh/install.omh
# BEGIN USER SETTINGS
# Directory where CppAD include files are located
# cppad_include_dir        = [ '/usr/include' ]
cppad_include_dir        = [ os.environ['HOME'] + '/prefix/cppad/include' ]
#
# Directory where Boost Python include files are located
boost_python_include_dir = [ '/usr/include' ]
#
# Director whre Boost Python library is located
# boost_python_lib_dir     = [ '/usr/lib' ] 
boost_python_lib_dir     = [ '/usr/lib64' ] 
#
# Name of the Boost Python library in boost_python_lib_dir.
# boost_python_lib         = [ 'boost_python-mt' ]
boost_python_lib         = [ 'boost_python' ]
# END USER SETTINGS
# ---------------------------------------------------------------------
# See http://docs.python.org/distutils/setupscript.html
# for documentation on how to write the script setup.py
#
# See  http://docs.python.org/install/index.html
# for documentation on how to use the script setup.py 
# ---------------------------------------------------------------------
# Values in setup.py that are replaced by build.sh
package_version    = '20140710',
# ---------------------------------------------------------------------
def remove_duplicates(alist) :
	n = len(alist)
	if n >= 2 :
		previous = alist[-1]
		for i in range(n-2, -1, -1 ) :
			if previous == alist[i] :
				del alist[i]
			else :
				previous = alist[i]
	return alist
# ---------------------------------------------------------------------
try:
	doc_files    = os.listdir('doc')
	for i in range( len(doc_files) ):
		doc_files[i] = os.path.join('doc', doc_files[i])
except:
	doc_files = []
#
directory              = os.path.join('share', 'doc')
directory              = os.path.join(directory, 'pycppad')
package_data_files     = [ (directory, doc_files ) ]
#
cppad_extension_name           = os.path.join('pycppad', 'cppad_')
cppad_extension_include_dirs   = get_numpy_include_dirs()
cppad_extension_include_dirs  += cppad_include_dir
cppad_extension_include_dirs  += boost_python_include_dir
remove_duplicates(cppad_extension_include_dirs)
cppad_extension_library_dirs   = boost_python_lib_dir
cppad_extension_libraries      = boost_python_lib
#
file_list = [ 'adfun.cpp', 'pycppad.cpp', 'vec2array.cpp', 'vector.cpp' ]
cppad_extension_sources = [ os.path.join('pycppad', f) for f in file_list ]
extension_modules = [ Extension( 
	cppad_extension_name                        , 
	cppad_extension_sources                     ,
	include_dirs = cppad_extension_include_dirs ,
	library_dirs = cppad_extension_library_dirs ,
	libraries    = cppad_extension_libraries    ,
) ]
#
brad_email        = 'bradbell @ seanet dot com'
sebastian_email   = 'sebastian dot walter @ gmail dot com'
setup(
	name         = 'pycppad',
	version      = '20140710',
	license      = 'BSD',
	description  = 'Python Algorihtmic Differentiation Using CppAD',
	author       = 'Bradley M. Bell and Sebastian F. Walter',
	author_email = sebastian_email + ' , ' + brad_email,
	url          = 'http://github.com/b45ch1/pycppad/tree/master',
	ext_modules  = extension_modules,
	packages     = [ 'pycppad'],
	package_dir  = { 'pycppad' : 'pycppad' },
	data_files   = package_data_files
)
