#! /bin/bash
# 
python_version="2.5"
python_config_dir="/usr/include/python$python_version"
if [ -e "$HOME/CppAD/trunk/cppad/cppad.hpp" ]
then
	cppad_dir="$HOME/CppAD/trunk"
else if [ -e "$HOME/workspace/pycppad/cppad-20081128/cppad.hpp" ]
then
	cppad_dir="/home/walter/workspace/pycppad/cppad-20081128"
else
	echo "Cannot find cppad/cppad.hpp"
	exit 1
fi
fi
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
# include <cassert>

typedef CppAD::AD<double>      ad_double;

namespace {
	using boost::python::object;

	// -------------------------------------------------------------
	class vec_ad_double {
	private:
		size_t  length_;
		object* obj_ptr_;
	public:
		typedef ad_double value_type;

		vec_ad_double(size_t length, object* obj_ptr) 
		: length_(length), obj_ptr_(obj_ptr)
		{ }

		size_t size(void)
		{	return length_; }

		ad_double& operator[](size_t i)
		{	assert( i < length_ );
			using boost::python::extract;
			return extract<ad_double&>(obj_ptr_[i])();
		}
	};
	// -------------------------------------------------------------
	void independent(boost::python::numeric::array &py_array)
	{	int* dims_ptr = PyArray_DIMS(py_array.ptr());
		int ndim      = PyArray_NDIM(py_array.ptr());
		int len       = dims_ptr[0];

		assert( ndim == 1 );
		assert( len >= 0 );

		// construct the vec_ad_doujble object corresponding to array
		size_t length = static_cast<size_t>(len);
		object* obj_ptr = 
			static_cast<object*>( PyArray_DATA(py_array.ptr()) );
		vec_ad_double vec(length, obj_ptr);

		CppAD::Independent(vec);

		return;
	}

}

BOOST_PYTHON_MODULE(python_cppad)
{
	using namespace boost::python;

	// some kind of hack to get numpy working  ---------------------------
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	// --------------------------------------------------------------------
	
	class_<ad_double>("ad_double", init<double>())
		.def(self_ns::str(self))
		.def(self * self)
	;
	def("independent", &independent);

}
EOF
# -------------------------------------------------------------------
echo "# Create the file python_cppad.tmp"
cat << EOF > python_cppad.py
from python_cppad import *
from numpy import array
x = array( [ ad_double(2) , ad_double(3) ] )
independent(x);
z = x[0] * x[1];
print 'x            = ',x
print 'x[0] * x[1]  = ',z
EOF
# -------------------------------------------------------------------
echo "# Compile python_cppad.cpp -------------------------------------------" 
#
cmd="g++ -fpic -g -c -Wall -I $python_config_dir -I $cppad_dir python_cppad.cpp"
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
