#! /bin/bash


# -------------------------------------------------------------------
python_config_dir="/usr/include/python2.5"
numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"


# -------------------------------------------------------------------
if [ ! -e "$python_config_dir/pyconfig.h" ]
then
echo "Must change python_config_dir or python_version in boost_python_and_numpy_tutorial.sh"
	exit 1
fi

# -------------------------------------------------------------------
echo "# Create the file boost_python_and_numpy_tutorial.cpp"
cat << EOF > boost_python_and_numpy_tutorial.cpp
#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

# include <boost/python.hpp>
# include <numpy/noprefix.h>
# include <numeric>
# include <iostream>
# include <string>
# include <cassert>

namespace {
	using boost::python::object;
	using namespace std;

	class ad_double{
		public:
		double _x;
		ad_double(double x) {_x = x;}
		ad_double& operator*=(const ad_double& rhs){
			(*this)._x *= rhs._x;
			return *this;
		}
		
	};

	ostream& operator << (ostream& os, const ad_double& s){
		os<<"adouble("<<s._x<<")";
	}

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
		/* type check */
		if(PyArray_TYPES(PyArray_TYPE(bpn_x.ptr())) != PyArray_OBJECT){
			PyErr_SetString(PyExc_ValueError, "This function can take only arrays of objects!");
			boost::python::throw_error_already_set();
		}
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
}

BOOST_PYTHON_MODULE(boost_python_and_numpy_tutorial)
{
	using namespace boost::python;
	using boost::python::self_ns::str;

	// some kind of hack to get numpy working  ---------------------------
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	class_<ad_double>("ad_double", init<double>())
		.def(str(self))
	;

	def("my_factorial", &my_factorial);
	def("square_elements", &square_elements);
}
EOF
# -------------------------------------------------------------------
echo "# Create the file boost_python_and_numpy_tutorial.tmp"
cat << EOF > boost_python_and_numpy_tutorial.py
from boost_python_and_numpy_tutorial import *
from numpy import array

print 'testing double arrays'
y = array([3.,5.],dtype=float)
z = square_elements(y)
print 'y=',y
print 'z=',z

print 'testing ad_double arrays'
ax = array([ad_double(3.5), ad_double(4.)])
ay = my_factorial(ax)
print 'ay=',ay

print 'testing error handling'
x =  array([3.,5.],dtype=float)
y = my_factorial(x)


EOF
# -------------------------------------------------------------------
echo "# Compile boost_python_and_numpy_tutorial.cpp -------------------------------------------"
#
cmd="g++ -fpic -g -c -Wall -I $python_config_dir -I $numpy_dir  boost_python_and_numpy_tutorial.cpp"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
# -------------------------------------------------------------------
echo "# Create boost_python_and_numpy_tutorial.so dynamic link library -----------------------"
# needed to link boost python
library_flags="-lboost_python"
cmd="g++ -shared -Wl,-soname,libboost_python_and_numpy_tutorial.so.1 $library_flags"
cmd="$cmd -o libboost_python_and_numpy_tutorial.so.1.0 boost_python_and_numpy_tutorial.o -lc"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
if [ -e boost_python_and_numpy_tutorial.so ]
then
	cmd="rm boost_python_and_numpy_tutorial.so"
	echo $cmd
	$cmd
fi
cmd="ln libboost_python_and_numpy_tutorial.so.1.0 boost_python_and_numpy_tutorial.so"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
# -------------------------------------------------------------------
echo "# Run boost_python_and_numpy_tutorial.py -----------------------------------------------"
cmd="python boost_python_and_numpy_tutorial.py"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
