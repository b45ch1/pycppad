#! /bin/bash
# 
python_version="2.5"
python_config_dir="/usr/include/python$python_version"

if [ "$USER" == "bradbell" ]
then
	cppad_dir="$HOME/CppAD/trunk"
	numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"
else if [ -e "$HOME/workspace/pycppad/cppad-20081128/cppad/cppad.hpp" ]
then
	numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"
	cppad_dir="../cppad-20081128"
else if [ -e "/u/walter/workspace/PyCPPAD/cppad-20081128/cppad/cppad.hpp" ]
then
	echo "this is wronski"
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
echo "# Create the file python_cppad.hpp"
cat << EOF > python_cppad.hpp
# include <cppad/cppad.hpp>
# include <boost/python.hpp>
# include <numpy/noprefix.h>
# include <numeric>
# include <iostream>
# include <string>
# include <cassert>

typedef CppAD::AD<double>      AD_double;

namespace {
	using std::cout;
	using std::endl;
	using boost::python::handle;
	using boost::python::object;
	using boost::python::numeric::array;
	// -------------------------------------------------------------
	class AD_double_vec {
	private:
		size_t       length_; // set by constructor only
		AD_double  *pointer_; // set by constructor only
		AD_double **handle_;  // set by constructor only
	public:
		typedef AD_double value_type;

		// constructor from a python array
		AD_double_vec(array& py_array);

		// constructor from size
		AD_double_vec(size_t length);

		// copy constructor
		AD_double_vec(const AD_double_vec& vec);

		// default constructor
		AD_double_vec(void);

		// destructor
		~AD_double_vec(void);

		// assignment operator
		void operator=(const AD_double_vec& vec);

		// size member function
		size_t size(void) const;

		// resize member function
		void resize(size_t length);

		// non constant element access
		AD_double& operator[](size_t i);

		// constant element access
		const AD_double& operator[](size_t i) const;
	};
	// -------------------------------------------------------------
	class double_vec {
	private:
		size_t    length_;  // set by constructor only
		double  *pointer_;  // set by constructor only
		bool    allocated_; // set by constructor only
	public:
		typedef double value_type;

		// constructor from a python array
		double_vec(array& py_array);

		// constructor from size
		double_vec(size_t length);

		// copy constructor
		double_vec(const double_vec& vec);

		// default constructor
		double_vec(void);

		// destructor
		~double_vec(void);

		// assignment operator
		void operator=(const double_vec& vec);

		// size member function
		size_t size(void) const;

		// resize member function
		void resize(size_t length);

		// non constant element access
		double& operator[](size_t i);

		// constant element access
		const double& operator[](size_t i) const;
	};
	// -------------------------------------------------------------
	array vector2array(const double_vec& vec);
	// -------------------------------------------------------------
	void independent(array& x_array);
	// -------------------------------------------------------------
	class ADFun_double {
	private:
		CppAD::ADFun<double> f_;
	public:
		// constructor
		ADFun_double(array& x_array, array& y_array);
		// Forwoard member function
		array Forward(int p, array& xp);
	};
	// -------------------------------------------------------------
}
EOF
# -------------------------------------------------------------------
echo "# Create the file python_cppad.cpp"
cat << EOF > python_cppad.cpp
# include "python_cppad.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

namespace {
	// -------------------------------------------------------------
	// class AD_double_vec
	//
	// constructor from a python array
	AD_double_vec::AD_double_vec(array& py_array)
	{
		// get array info
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

		// set private data
		using boost::python::extract;
		length_  = static_cast<size_t>(length);
		pointer_ = 0;
		handle_  = CPPAD_TRACK_NEW_VEC(length_, handle_);
		for(size_t i = 0; i < length_; i++) handle_[i] = 
			& extract<AD_double&>(obj_ptr[i])(); 
		return;
	}
	// constructor from size
	AD_double_vec::AD_double_vec(size_t length)
	{
		length_  = length;
		pointer_ = CPPAD_TRACK_NEW_VEC(length, pointer_);
		handle_  = CPPAD_TRACK_NEW_VEC(length, handle_);
		for(size_t i = 0; i < length_; i++)
			handle_[i] = pointer_ + i;
		return;
	}

	// copy constructor
	AD_double_vec::AD_double_vec(const AD_double_vec& vec)
	{
		length_   = vec.length_;
		pointer_  = CPPAD_TRACK_NEW_VEC(length_, pointer_);
		handle_   = CPPAD_TRACK_NEW_VEC(length_, handle_);
		for(size_t i = 0; i < length_; i++)
		{	handle_[i]  = pointer_ + i;
			pointer_[i] = vec[i];
		}
	}

	// default constructor
	AD_double_vec::AD_double_vec(void)
	{
		length_  = 0;
		pointer_ = 0;
		handle_  = 0;
	}

	// destructor
	AD_double_vec::~AD_double_vec(void)
	{
		if( handle_ != 0 )
			CPPAD_TRACK_DEL_VEC(handle_); 
		if( pointer_ != 0 )
			CPPAD_TRACK_DEL_VEC(pointer_);	
	}

	// assignment operator
	void AD_double_vec::operator=(const AD_double_vec& vec)
	{
		assert( length_ == vec.length_ ); 
		for(size_t i = 0; i < length_; i++)
			*handle_[i] = *(vec.handle_[i]);
		return;
	}

	// size member function
	size_t AD_double_vec::size(void) const
	{	return length_; }

	// resize 
	void AD_double_vec::resize(size_t length)
	{
		if( handle_ != 0 )
			CPPAD_TRACK_DEL_VEC(handle_);
		if( pointer_ != 0 )
			CPPAD_TRACK_DEL_VEC(pointer_);
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		handle_    = CPPAD_TRACK_NEW_VEC(length, handle_);
		length_    = length;
		for(size_t i = 0; i < length_; i++)
			handle_[i]  = pointer_ + i;
	}

	// non constant element access
	AD_double& AD_double_vec::operator[](size_t i)
	{	assert( i < length_ );
		return *handle_[i];
	}

	// constant element access
	const AD_double& AD_double_vec::operator[](size_t i) const
	{	assert( i < length_ );
		return *handle_[i];
	}
	// -------------------------------------------------------------
	// class double_vec
	//
	// constructor from a python array
	double_vec::double_vec(array& py_array)
	{	// get array info
		int* dims_ptr = PyArray_DIMS(py_array.ptr());
		int ndim      = PyArray_NDIM(py_array.ptr());
		int length    = dims_ptr[0];

		// check array info
		assert( ndim == 1 );
		assert( length >= 0 );

		// set private data
		length_    = static_cast<size_t>( length );
		pointer_   = static_cast<double*>( 
			PyArray_DATA(py_array.ptr()) 
		);
		allocated_ = false;
		return;
	}

	// constructor from size
	double_vec::double_vec(size_t length)
	{	// set private data
		length_    = length;
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		allocated_ = true;
		return;
	}

	// copy constructor
	double_vec::double_vec(const double_vec& vec)
	{	length_    = vec.length_;
		pointer_   = CPPAD_TRACK_NEW_VEC(length_, pointer_);
		allocated_ = true;
		for(size_t i = 0; i < length_; i++)
			pointer_[i] = vec[i];
	}

	// default constructor
	double_vec::double_vec(void)
	{	length_    = 0;
		pointer_   = 0;
		allocated_ = false;
	}

	// destructor
	double_vec::~double_vec(void)
	{	if( allocated_ )
			CPPAD_TRACK_DEL_VEC(pointer_);	
	}

	// assignment operator
	void double_vec::operator=(const double_vec& vec)
	{	assert( length_ == vec.length_ ); 
		for(size_t i = 0; i < length_; i++)
			pointer_[i] = vec.pointer_[i];
		return;
	}

	// size member function
	size_t double_vec::size(void) const
	{	return length_; }

	// resize 
	void double_vec::resize(size_t length)
	{	if( allocated_ )
			CPPAD_TRACK_DEL_VEC(pointer_);
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		length_    = length;
		allocated_ = true;
	}

	// non constant element access
	double& double_vec::operator[](size_t i)
	{	assert( i < length_ );
		return pointer_[i];
	}

	// constant element access
	const double& double_vec::operator[](size_t i) const
	{	assert( i < length_ );
		return pointer_[i];
	}
	// -------------------------------------------------------------
	array vector2array(const double_vec& vec)
	{	int n = static_cast<int>( vec.size() );
		assert( n >= 0 );

		object obj(handle<>( PyArray_FromDims(1, &n, PyArray_DOUBLE) ));
		// for some unknown reason,
		// static_cast<PyArrayObject*> ( obj.ptr() ) does not work ?
		double *ptr = static_cast<double*> ( PyArray_DATA (
			(PyArrayObject*) ( obj.ptr() )
		));
		for(size_t i = 0; i < vec.size(); i++){
			ptr[i] = vec[i];
		}
		return  static_cast<array>( obj );
	}
	// -------------------------------------------------------------
	void independent(array& x_array)
	{	AD_double_vec x_vec(x_array);
		CppAD::Independent(x_vec);
		return;
	}
	// -------------------------------------------------------------
	// class ADFun_double
	//
	ADFun_double::ADFun_double(array& x_array, array& y_array)
	{	AD_double_vec x_vec(x_array);
		AD_double_vec y_vec(y_array);

		f_.Dependent(x_vec, y_vec);
	}

	array ADFun_double::Forward(int p, array& xp)
	{	size_t     p_sz(p);
		double_vec xp_vec(xp);
		double_vec result = f_.Forward(p_sz, xp_vec);
		return vector2array(result);
	}
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
	
	class_<AD_double>("AD_double", init<double>())
		.def(str(self))
		.def(self * self)
	;
	def("independent", &independent);

	class_<ADFun_double>("ADFun_double", init< array& , array& >())
		.def("Forward", &ADFun_double::Forward)
	;
}
EOF
# -------------------------------------------------------------------
echo "# Create the file python_cppad.py"
cat << EOF > python_cppad.tmp
from python_cppad import *
from numpy import array
x = array( [ AD_double(2) , AD_double(3) ] )
independent(x);
y = array( [ x[0] * x[1] ] );
f = ADFun_double(x, y)
print 'f(x0, x1) = x0 * x1'
p  = 0
xp = array( [ 3. , 4. ] )
fp = f.Forward(p, xp)
print 'f(3, 4)            = ', fp
p = 1
xp = array( [ 1. , 0. ] )
fp = f.Forward(p, xp)
print 'partial_x0 f(3, 4) = ' , fp
xp = array( [ 0. , 1. ] )
fp = f.Forward(p, xp)
print 'partial_x1 f(3, 4) = ' , fp
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
