/*
---------------------------------------------------------------------------
$begin abs$$ $newlinech #$$
$spell
$$

$section Absolute Value Functions$$

$head Syntax$$
$icode%y% = abs(%x%)%$$

$head Purpose$$
Sets $icode y$$ equal to the absolute value of $latex x$$.

$head x$$
The argument $icode x$$ can be an instance of $code float$$,
an $code a_float$$, an $code a2float$$, or an $cref/array/$$ of such objects.

$head y$$
If $icode x$$ is an instance of $code float$$,
$icode y$$ will also be an instance of $icode float$$.
Otherwise $icode y$$ will have the same type as $icode x$$.
$pre

$$
In the case where $icode x$$ is an array, $icode y$$ will 
the same shape as $icode x$$ and the elements of $icode y$$
will have the  same type as the elements of $icode x$$.

$children%
	example/abs.py
%$$
$head Example$$
The file $cref/abs.py/$$ 
contains an example and test of these functions.

$end
---------------------------------------------------------------------------
*/
# include "setup.hpp"
# include "vector.hpp"
# include "vec2array.hpp"
# include "adfun.hpp"

# define PY_ARRAY_UNIQUE_SYMBOL PyArray_Pycppad

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

# define PYCPPAD_UNARY_FUNCTION(Name, Base)                              \
     CppAD::AD<Base> (* Name##_AD_##Base ) (const CppAD::AD<Base> &x ) = \
          CppAD::Name<Base>; 

# define PYCPPAD_STD_MATH_LINK_CPP(Base)   \
     PYCPPAD_UNARY_FUNCTION(acos,  Base)   \
     PYCPPAD_UNARY_FUNCTION(asin,  Base)   \
     PYCPPAD_UNARY_FUNCTION(atan,  Base)   \
     PYCPPAD_UNARY_FUNCTION(cos,   Base)   \
     PYCPPAD_UNARY_FUNCTION(cosh,  Base)   \
     PYCPPAD_UNARY_FUNCTION(exp,   Base)   \
     PYCPPAD_UNARY_FUNCTION(log,   Base)   \
     PYCPPAD_UNARY_FUNCTION(log10, Base)   \
     PYCPPAD_UNARY_FUNCTION(sin,   Base)   \
     PYCPPAD_UNARY_FUNCTION(sinh,  Base)   \
     PYCPPAD_UNARY_FUNCTION(sqrt,  Base)   \
     PYCPPAD_UNARY_FUNCTION(tan,   Base)   \
     PYCPPAD_UNARY_FUNCTION(tanh,  Base) 

# define PYCPPAD_POW_LINK_CPP(Base)                                         \
     CppAD::AD<Base> (* pow_AD_##Base##_AD_##Base)                          \
     (const CppAD::AD<Base> &x, const CppAD::AD<Base> & y ) = &CppAD::pow;  \
     CppAD::AD<Base> (* pow_double_AD_##Base)                               \
     (const double &x, const CppAD::AD<Base> &y ) = &CppAD::pow;            \
     CppAD::AD<Base> (* pow_AD_##Base##_double)                             \
     (const CppAD::AD<Base> &x, const double &y ) = &CppAD::pow;            \
     CppAD::AD<Base> (* pow_int_AD_##Base)                                  \
     (const int &x, const CppAD::AD<Base> &y ) = &CppAD::pow;               \
     CppAD::AD<Base> (* pow_AD_##Base##_int)                                \
     (const CppAD::AD<Base> &x, const int &y ) = &CppAD::pow;

# define PYCPPAD_STD_MATH_LINK_PY(Base)  \
     .def("arccos",  acos_AD_##Base)     \
     .def("arcsin",  asin_AD_##Base)     \
     .def("arctan",  atan_AD_##Base)     \
     .def("cos",     cos_AD_##Base)      \
     .def("cosh",    cosh_AD_##Base)     \
     .def("exp",     exp_AD_##Base)      \
     .def("log",     log_AD_##Base)      \
     .def("log10",   log10_AD_##Base)    \
     .def("sin",     sin_AD_##Base)      \
     .def("sinh",    sinh_AD_##Base)     \
     .def("sqrt",    sqrt_AD_##Base)     \
     .def("tan",     tan_AD_##Base)      \
     .def("tanh",    tanh_AD_##Base)

# define PYCPPAD_POW_LINK_PY(Base)              \
     .def("__pow__", pow_AD_##Base##_AD_##Base) \
     .def("__pow__", pow_double_AD_##Base)      \
     .def("__pow__", pow_AD_##Base##_double)    \
     .def("__pow__", pow_int_AD_##Base)         \
     .def("__pow__", pow_AD_##Base##_int)

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
	// Kludge: Pass level to Independent until we know how to determine if 
	// the elements are x_array are AD_double or AD_AD_double.
	array Independent(array& x_array, int level)
	{	PYCPPAD_ASSERT( 
			level == 1 || level == 2,
			"independent: level argument must be 1 or 2."
		);
		if( level == 1 )
		{
			double_vec      x(x_array);
			AD_double_vec a_x(x.size() );
			for(size_t j = 0; j < x.size(); j++)
				a_x[j] = x[j];
			CppAD::Independent(a_x);
			return vec2array(a_x);
		}
		AD_double_vec      x(x_array);
		AD_AD_double_vec a_x(x.size() );
		for(size_t j = 0; j < x.size(); j++)
			a_x[j] = x[j];
		CppAD::Independent(a_x);
		return vec2array(a_x);
	}
	// -------------------------------------------------------------
	double double_(const AD_double& x)
	{	return Value(x); }
	AD_double AD_double_(const AD_AD_double& x)
	{	return Value(x); }
}

BOOST_PYTHON_MODULE(pycppad)
{
	// AD_double is used in pycppad namespace
	typedef CppAD::AD<double>    AD_double;
	typedef CppAD::AD<AD_double> AD_AD_double;

	PYCPPAD_STD_MATH_LINK_CPP(double);
	PYCPPAD_POW_LINK_CPP(double);
	PYCPPAD_UNARY_FUNCTION(abs, double);

	PYCPPAD_STD_MATH_LINK_CPP(AD_double);
	PYCPPAD_POW_LINK_CPP(AD_double);
	PYCPPAD_UNARY_FUNCTION(abs, AD_double);

	// here are the things we are using from boost::python
	using boost::python::numeric::array;
	using boost::python::class_;
	using boost::python::init;
	using boost::python::self;
	using boost::python::self_ns::str;
	using boost::python::def;

	using pycppad::ADFun_double;
	using pycppad::ADFun_AD_double;

	// some kind of hack to get numpy working  ---------------------------
	import_array(); 
	pycppad::vec2array_import_array();
	array::set_module_and_type("numpy", "ndarray");
	// --------------------------------------------------------------------
	def("independent", pycppad::Independent);
	def("float_",     pycppad::double_);
	def("a_float_",   pycppad::AD_double_);
	// --------------------------------------------------------------------
	class_<AD_double>("a_float", init<double>())
		.def(str(self))
		PYCPPAD_OPERATOR_LIST
		PYCPPAD_STD_MATH_LINK_PY(double)
		PYCPPAD_POW_LINK_PY(double)
		.def("__abs__",  abs_AD_double)
	;

	class_<ADFun_double>("adfun_float", init< array& , array& >())
		.def("domain",    &ADFun_double::Domain)
		.def("range",     &ADFun_double::Range)
		.def("forward",   &ADFun_double::Forward)
		.def("reverse",   &ADFun_double::Reverse)
		.def("jacobian_", &ADFun_double::Jacobian)
	;
	// --------------------------------------------------------------------
	class_<AD_AD_double>("a2float", init<AD_double>())
		.def(str(self))
		PYCPPAD_OPERATOR_LIST
		PYCPPAD_STD_MATH_LINK_PY(AD_double)
		PYCPPAD_POW_LINK_PY(AD_double)
		.def("__abs__",  abs_AD_AD_double)
	;
	class_<ADFun_AD_double>("adfun_a_float", init< array& , array& >())
		.def("domain",    &ADFun_AD_double::Domain)
		.def("range",     &ADFun_AD_double::Range)
		.def("forward",   &ADFun_AD_double::Forward)
		.def("reverse",   &ADFun_AD_double::Reverse)
		.def("jacobian_", &ADFun_AD_double::Jacobian)
	;
}
