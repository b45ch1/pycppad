# include "setup.hpp"
# include "vector.hpp"
# include "adfun.hpp"

# define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

# define PYCPPAD_BINARY(op)       \
     .def(self     op self)       \
     .def(double() op self)       \
     .def(self     op double())


# define PYCPPAD_OPERATOR_LIST \
                               \
     PYCPPAD_BINARY(+)         \
     PYCPPAD_BINARY(-)         \
     PYCPPAD_BINARY(*)         \
     PYCPPAD_BINARY(/)         \
                               \
     PYCPPAD_BINARY(<)         \
     PYCPPAD_BINARY(>)         \
     PYCPPAD_BINARY(<=)        \
     PYCPPAD_BINARY(>=)        \
     PYCPPAD_BINARY(==)        \
     PYCPPAD_BINARY(!=)        \
                               \
     .def(self += self)        \
     .def(self -= self)        \
     .def(self *= self)        \
     .def(self /= self)        \
                               \
     .def(self += double())    \
     .def(self -= double())    \
     .def(self *= double())    \
     .def(self /= double()) 

// Kludge: STD_MATH is only working as a member function (so far)
// (so have only implemented on function for testing new ideas)
# define PYCPPAD_STD_MATH_LINK(Base)                                \
     CppAD::AD<Base> (* sin_AD_##Base) (const CppAD::AD<Base> &x) = \
          &CppAD::sin<Base>;

# define PYCPPAD_STD_MATH_LIST(Base)   \
     .def("sin",  sin_AD_##Base)

namespace pycppad {
	// Replacement for the CppAD error handler
	void error_handler(
		bool known           ,
		int  line            ,
		const char *file     ,
		const char *exp      ,
		const char *msg      )
	{	if( ! known ) msg = 
			"Bug detected in pycppad, Please report this.";

		PyErr_SetString(PyExc_ValueError, msg);
		boost::python::throw_error_already_set();
		// erorr handler must not return
	}
	// This ojbect lasts forever, so forever replacement of 
	// the default CppAD erorr handler 
	CppAD::ErrorHandler myhandler(error_handler);
	// -------------------------------------------------------------
	// Kludge: The vector2array functions should be in vector.cpp 
	// and vector.hpp, but it seems they also have to me in the same 
	// file as # define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle
	//  
	array vector2array(const double_vec& vec)
	{	int n = static_cast<int>( vec.size() );
		PYCPPAD_ASSERT( n >= 0 , "");

		object obj(handle<>( PyArray_FromDims(1, &n, PyArray_DOUBLE) ));
		double *ptr = static_cast<double*> ( PyArray_DATA (
			reinterpret_cast<PyArrayObject*> ( obj.ptr() )
		));
		for(size_t i = 0; i < vec.size(); i++){
			ptr[i] = vec[i];
		}
		return  static_cast<array>( obj );
	}
	array vector2array(const AD_double_vec& vec)
	{	int n = static_cast<int>( vec.size() );
		PYCPPAD_ASSERT( n >= 0 , "");

		object obj(handle<>( PyArray_FromDims(1, &n, PyArray_OBJECT) ));
		for(size_t i = 0; i < vec.size(); i++){
			obj[i] = vec[i];
		}
		return  static_cast<array>( obj );
	}
	array vector2array(size_t m, size_t n, const double_vec& vec)
	{
		PYCPPAD_ASSERT(m * n == vec.size(), "");

		int dims[2];
		dims[0] = static_cast<int>(m);
		dims[1] = static_cast<int>(n);
		PYCPPAD_ASSERT( dims[0] >= 0, "");
		PYCPPAD_ASSERT( dims[1] >= 0, "");
		object obj(handle<>( 
			PyArray_FromDims(2, dims, PyArray_DOUBLE) 
		));
		double *ptr = static_cast<double*> ( PyArray_DATA (
			reinterpret_cast<PyArrayObject*> ( obj.ptr() )
		));
		size_t i = vec.size();
		while(i--)
			ptr[i] = vec[i];
		return  static_cast<array>( obj );
	}
	array vector2array(size_t m, size_t n, const AD_double_vec& vec)
	{
		PYCPPAD_ASSERT(m * n == vec.size(), "");

		int dims[2];
		dims[0] = static_cast<int>(m);
		dims[1] = static_cast<int>(n);
		PYCPPAD_ASSERT( dims[0] >= 0, "");
		PYCPPAD_ASSERT( dims[1] >= 0, "");
		object obj(handle<>( 
			PyArray_FromDims(2, dims, PyArray_OBJECT) 
		));
		size_t i = vec.size();
		while(i--)
			obj[i] = vec[i];
		return  static_cast<array>( obj );
	}
	// -------------------------------------------------------------
	// Kludge: Pass level to Independent until we know how to determine if 
	// the elements are x_array are AD_double or AD_AD_double.
	void Independent(array& x_array, int level)
	{	PYCPPAD_ASSERT( 
			level == 1 || level == 2,
			"independent: level argument must be 1 or 2."
		);
		if( level == 1 )
		{	AD_double_vec x_vec(x_array);
			CppAD::Independent(x_vec);
		}
		else
		{	AD_AD_double_vec x_vec(x_array);
			CppAD::Independent(x_vec);
		}
	}
}

BOOST_PYTHON_MODULE(pycppad)
{
	// AD_double is used in pycppad namespace
	typedef CppAD::AD<double>    AD_double;
	typedef CppAD::AD<AD_double> AD_AD_double;

	PYCPPAD_STD_MATH_LINK(double);
	PYCPPAD_STD_MATH_LINK(AD_double);

	// here are the things we are using from boost::python
	using boost::python::numeric::array;
	using boost::python::class_;
	using boost::python::init;
	using boost::python::self;
	using boost::python::self_ns::str;

	using pycppad::Independent;
	using pycppad::ADFun_double;
	using pycppad::ADFun_AD_double;

	// some kind of hack to get numpy working  ---------------------------
	import_array(); 
	array::set_module_and_type("numpy", "ndarray");
	// --------------------------------------------------------------------
	def("independent", &Independent);
	// --------------------------------------------------------------------
	class_<AD_double>("a_double", init<double>())
		.def(str(self))
		PYCPPAD_OPERATOR_LIST
		PYCPPAD_STD_MATH_LIST(double)
	;

	class_<ADFun_double>("adfun_double", init< array& , array& >())
		.def("forward", &ADFun_double::Forward)
		.def("reverse", &ADFun_double::Reverse)
		.def("jacobian", &ADFun_double::Jacobian)
	;
	// --------------------------------------------------------------------
	class_<AD_AD_double>("a2double", init<AD_double>())
		.def(str(self))
		PYCPPAD_OPERATOR_LIST
		PYCPPAD_STD_MATH_LIST(AD_double)
	;
	class_<ADFun_AD_double>("adfun_a_double", init< array& , array& >())
		.def("forward", &ADFun_AD_double::Forward)
		.def("reverse", &ADFun_AD_double::Reverse)
		.def("jacobian", &ADFun_AD_double::Jacobian)
	;
}
