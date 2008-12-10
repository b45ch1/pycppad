#! /bin/bash
# 
python_version="2.5"
python_config_dir="/usr/include/python$python_version"
numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"
cppad_dir="../cppad-20081128"
# -------------------------------------------------------------------
if [ ! -e "$python_config_dir/pyconfig.h" ]
then
echo "Must change python_config_dir or python_version in python_cppad.sh"
	exit 1
fi
python --version >& python_cppad.tmp
py_version=`cat python_cppad.tmp`
if ! grep "Python $python_version" python_cppad.tmp > /dev/null
then
	echo "Must change python_version in python_cppad.sh"
	exit 1
fi
# -------------------------------------------------------------------
echo "# Create the file python_cppad.cpp"
cat << EOF > python_cppad.cpp
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

# include <cppad/cppad.hpp>
# include <boost/python.hpp>
# include <numpy/noprefix.h>
# include <numeric>
# include <iostream>
# include <string>

using namespace std;

typedef CppAD::AD<double>        ad_double;


ad_double my_factorial(boost::python::numeric::array &bpn_x){
	int* dims_ptr = PyArray_DIMS(bpn_x.ptr());
	int ndim = PyArray_NDIM(bpn_x.ptr());
	int N = dims_ptr[0];
	cout<<"Message from C++:"<<endl;
	cout<<"The received array is a "<<ndim<<" dimensional array"<<endl;
	cout<<"The size of the first dimension is "<<N<<endl;
	
	/* get a pointer to the an array of Python objects from bpn_x */
	boost::python::object* obj_x = (boost::python::object*) PyArray_DATA(bpn_x.ptr());
	
	/* get C++ object out of the Python object with extract
	and multiply them together */
	ad_double tmp(1.);
	
	for(int n = 0; n != N; ++n){
		tmp *= boost::python::extract<ad_double&>(obj_x[n])();
	}
	
	return tmp;
	
	
}

BOOST_PYTHON_MODULE(python_cppad)
{
	using namespace boost::python;
	import_array();    /* some kind of hack to get numpy working */
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");   /* some kind of hack to get numpy working */
	
	def("my_factorial", &my_factorial);
	class_<ad_double>("ad_double", init<double>())
		.def(self_ns::str(self))
		.def(self * self)
	;

}
EOF
# -------------------------------------------------------------------
echo "# Create the file python_cppad.py"
cat << EOF > python_cppad.py
from python_cppad import *
from numpy import array
from scipy import factorial
z = array([ad_double(i) for i in range(1,13)])
print 'z=',z
w = my_factorial(z)
print 'my_factorial(z)=',w
print 'factorial(z)',factorial(12)
EOF
# -------------------------------------------------------------------
echo "# Compile python_cppad.cpp -------------------------------------------" 
#
cmd="g++ -fpic -g -c -Wall -I $python_config_dir -I $numpy_dir -I $cppad_dir python_cppad.cpp"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
# -------------------------------------------------------------------
echo "# Create python_cppad.so dynamic link library -----------------------"
# needed to link boost python
library_flags="-lboost_python"
cmd="g++ -shared -Wl,-soname,libpython_cppad.so.1 $library_flags"
cmd="$cmd -o libpython_cppad.so.1.0 python_cppad.o -lc"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
if [ -e python_cppad.so ]
then
	cmd="rm python_cppad.so"
	echo $cmd
	$cmd
fi
cmd="ln libpython_cppad.so.1.0 python_cppad.so"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
# -------------------------------------------------------------------
echo "# Run python_cppad.py -----------------------------------------------"
cmd="python python_cppad.py"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
