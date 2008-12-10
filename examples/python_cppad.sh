#! /bin/bash
# 
python_version="2.5"
python_config_dir="/usr/include/python$python_version"

if [ -e "$HOME/CppAD/trunk/cppad/cppad.hpp" ]
then
	cppad_dir="$HOME/CppAD/trunk"
else if [ -e "$HOME/workspace/pycppad/cppad-20081128/cppad/cppad.hpp" ]
then
	numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"
	cppad_dir="../cppad-20081128"
else if [ -e "/u/walter/workspace/PyCPPAD/cppad-20081128/cppad/cppad.hpp" ]
then
	numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"
	cppad_dir="/u/walter/workspace/PyCPPAD/cppad-20081128"
else
	echo "Cannot find cppad/cppad.hpp"
	exit 1
fi
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
	using namespace std;



	boost::python::numeric::array  square_elements(boost::python::numeric::array &bpn_x){
		int* dims_ptr = PyArray_DIMS(bpn_x.ptr());
		int ndim = PyArray_NDIM(bpn_x.ptr());
		int N = dims_ptr[0];
		/* get a pointer to the an array of Python objects from bpn_x */
		double* x = (double*) PyArray_DATA(bpn_x.ptr());


		boost::python::object y_obj(boost::python::handle<>(PyArray_FromDims(1, &N, PyArray_DOUBLE)));
		double *y = (double*) PyArray_DATA((PyArrayObject*) y_obj.ptr());
		
		for(int n = 0; n != N; ++n){
			y[n] = x[n]*x[n];
			x[n] += 13.;
		}
		
		return boost::python::extract<boost::python::numeric::array>(y_obj);
	}


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
	def("square_elements", &square_elements);
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

y = array([3.,5.],dtype=float)
z = square_elements(y)
print y,z
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
