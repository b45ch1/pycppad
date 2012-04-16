#!/usr/bin/env python
import os
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
# ---------------------------------------------------------------------
# The code below is included verbatim in omh/install.omh
# BEGIN USER SETTINGS
# Directory where CppAD include files are located
cppad_include_dir        = [ '/usr/include' ]
cppad_include_dir        = [ os.environ['HOME'] + '/prefix/cppad/include' ]
# Directory where Boost Python library and include files are located
boost_python_include_dir = [ '/usr/include' ]
boost_python_lib_dir     = [ '/usr/lib' ] 
# Name of the Boost Python library in boost_python_lib_dir.
boost_python_lib         = [ 'boost_python-mt' ]
# END USER SETTINGS
# ---------------------------------------------------------------------
# See http://docs.python.org/distutils/setupscript.html
# for documentation on how to write the script setup.py
#
# See  http://docs.python.org/install/index.html
# for documentation on how to use the script setup.py 
# ---------------------------------------------------------------------
# Values in setup.py that are replaced by build.sh
package_version    = '20120416'
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
cppad_extension_name           = 'pycppad' + '/cppad_'
cppad_extension_include_dirs   = get_numpy_include_dirs()
cppad_extension_include_dirs  += cppad_include_dir
cppad_extension_include_dirs  += boost_python_include_dir
remove_duplicates(cppad_extension_include_dirs)
cppad_extension_library_dirs   = boost_python_lib_dir
cppad_extension_libraries      = boost_python_lib
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
