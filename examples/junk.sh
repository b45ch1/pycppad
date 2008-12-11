#! /bin/bash
# 
python_version="2.5"
python_config_dir="/usr/include/python$python_version"
cppad_dir="../cppad-20081128"
numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"
# -------------------------------------------------------------------
if [ ! -e "$cppad_dir/cppad/cppad.hpp" ]
then
	echo "Cannot find cppad/cppad.hpp"
	exit 1
fi
# -------------------------------------------------------------------
if [ ! -d "$numpy_dir" ]
then
	echo "Must change numpy_dir in python_cppad.sh"
	exit 1
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
	using std::cout;
	using std::endl;
	using boost::python::handle;
	using boost::python::object;
	using boost::python::numeric::array;
	//
	// -------------------------------------------------------------
	template <class Scalar>
	class vector {
	private:
		size_t   length_;  // set by constructor only
		Scalar  *pointer_; // set by constructor only
		Scalar **handle_;  // set by constructor only
	public:
		typedef Scalar value_type;

		// constructor from a python array
		vector(array& py_array)
		{	// get array info
cout << "vector(array): begin" << endl;
			int* dims_ptr = PyArray_DIMS(py_array.ptr());
			int ndim      = PyArray_NDIM(py_array.ptr());
			int length    = dims_ptr[0];

			// check array info
			assert( ndim == 1 );
			assert( length >= 0 );

			// pointer to object
			object *obj_ptr = static_cast<object*>( 
				PyArray_DATA(py_array.ptr()) 
			);

cout << "vector(array): set private data" << endl;
			// set private data
			using boost::python::extract;
			length_  = static_cast<size_t>(length);
cout << "vector(array): length_ = " << length_ << endl;
			pointer_ = 0;
			handle_  = CPPAD_TRACK_NEW_VEC(length_, handle_);
cout << "vector(array): handle_ = " << handle_ << endl;
			for(size_t i = 0; i < length_; i++)
			{
cout << "vector(array): i = " << i << endl;
			Scalar w = extract<Scalar&>(obj_ptr[i]);
cout << "vector(array): array[i] = " << w <<  endl;
		
				handle_[i] = & extract<Scalar&>(obj_ptr[i])(); 
			}
cout << "vector(array): end" << endl;
			return;
		}

		// constructor from size
		vector(size_t length)
		{	// set private data
			length_  = length;
			pointer_ = CPPAD_TRACK_NEW_VEC(length, pointer_);
			for(size_t i = 0; i < length_; i++)
				handle_[i] = pointer_ + i;
			return;
		}

		// destructor
		~vector(void)
		{	CPPAD_TRACK_DEL_VEC(handle_); 
			if( pointer_ != 0 )
				CPPAD_TRACK_DEL_VEC(pointer_);	
		}

		// size member function
		size_t size(void) const
		{	return length_; }

		// non constant element access
		Scalar& operator[](size_t i)
		{	assert( i < length_ );
			return *handle_[i];
		}

		// constant element access
		const Scalar& operator[](size_t i) const
		{	assert( i < length_ );
			return *handle_[i];
		}
	};
	// -------------------------------------------------------------
	array vector2array(const vector<double>& vec)
	{	int n = static_cast<int>( vec.size() );
		assert( n >= 0 );

		object obj(handle<>( PyArray_FromDims(1, &n, PyArray_DOUBLE) ));
		// for some unknown reason,
		// static_cast<PyArrayObject*> ( obj.ptr() ) does not work ?
		double *ptr = static_cast<double*> ( PyArray_DATA (
			(PyArrayObject*) ( obj.ptr() )
		));
		for(size_t i = i; i < vec.size(); i++)
			ptr[i] = vec[i];
		return  static_cast<array>( obj );
	}
	// -------------------------------------------------------------
	void independent(array& x_array)
	{	vector<ad_double> x_vec(x_array);
		CppAD::Independent(x_vec);
		return;
	}
	// -------------------------------------------------------------
	class adfun_double {
	private:
		CppAD::ADFun<double> f_;
	public:
		adfun_double(array& x_array, array& y_array)
		{	vector<ad_double> x_vec(x_array);
			vector<ad_double> y_vec(y_array);

			f_.Dependent(x_vec, y_vec);
		}

		array Forward(int p, array& xp)
		{	cout << "Forward: begin" << endl;
			size_t         p_sz(p);
			vector<double> xp_vec(xp);
			cout << "Forward: f_Forward" << endl;
			vector<double> result = f_.Forward(p_sz, xp_vec);
			cout << "Forward: end" << endl;
			return vector2array(result);
		}
	};
	// -------------------------------------------------------------

}

BOOST_PYTHON_MODULE(python_cppad)
{
	// here are the things we are using from boost::python
	using boost::python::numeric::array;
	using boost::python::class_;
	using boost::python::init;
	using boost::python::self;
	using boost::python::self_ns::str;

	// some kind of hack to get numpy working  ---------------------------
	import_array(); 
	array::set_module_and_type("numpy", "ndarray");
	// --------------------------------------------------------------------
	
	class_<ad_double>("ad_double", init<double>())
		.def(str(self))
		.def(self * self)
	;
	def("independent", &independent);

	class_<adfun_double>("adfun_double", init< array& , array& >())
		.def("Forward", &adfun_double::Forward)
	;
}
EOF
# -------------------------------------------------------------------
echo "# Create the file python_cppad.tmp"
cat << EOF > python_cppad.py
from python_cppad import *
from numpy import array
x = array( [ ad_double(2) , ad_double(3) ] )
independent(x);
y = array( [ x[0] * x[1] ] );
f = adfun_double(x, y)
print '2 * 3 = ',y
p  = 0
xp = array( [ 3. , 4. ] )
fp = f.Forward(p, xp)
print '3 * 4 = ', fp
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
