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

# define PYCPPAD_STD_MATH_LINK(Base)                                   \
     CppAD::AD<Base> (* acos_AD_##Base) (const CppAD::AD<Base> &x) =   \
          CppAD::acos<Base>;                                           \
     CppAD::AD<Base> (* asin_AD_##Base) (const CppAD::AD<Base> &x) =   \
          CppAD::asin<Base>;                                           \
     CppAD::AD<Base> (* atan_AD_##Base) (const CppAD::AD<Base> &x) =   \
          CppAD::atan<Base>;                                           \
     CppAD::AD<Base> (* cos_AD_##Base) (const CppAD::AD<Base> &x) =    \
          CppAD::cos<Base>;                                            \
     CppAD::AD<Base> (* cosh_AD_##Base) (const CppAD::AD<Base> &x) =   \
          CppAD::cosh<Base>;                                           \
     CppAD::AD<Base> (* exp_AD_##Base) (const CppAD::AD<Base> &x) =    \
          CppAD::exp<Base>;                                            \
     CppAD::AD<Base> (* log_AD_##Base) (const CppAD::AD<Base> &x) =    \
          CppAD::log<Base>;                                            \
     CppAD::AD<Base> (* log10_AD_##Base) (const CppAD::AD<Base> &x) =  \
          CppAD::log10<Base>;                                          \
     CppAD::AD<Base> (* sin_AD_##Base) (const CppAD::AD<Base> &x) =    \
          CppAD::sin<Base>;                                            \
     CppAD::AD<Base> (* sinh_AD_##Base) (const CppAD::AD<Base> &x) =   \
          CppAD::sinh<Base>;                                           \
     CppAD::AD<Base> (* sqrt_AD_##Base) (const CppAD::AD<Base> &x) =   \
          CppAD::sqrt<Base>;                                           \
     CppAD::AD<Base> (* tan_AD_##Base) (const CppAD::AD<Base> &x) =    \
          CppAD::tan<Base>;                                            \
     CppAD::AD<Base> (* tanh_AD_##Base) (const CppAD::AD<Base> &x) =   \
          CppAD::tanh<Base>; 

# define PYCPPAD_STD_MATH_LIST(Base)     \
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
	pycppad::vec2array_import_array();
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
		.def("domain",    &ADFun_double::Domain)
		.def("range",     &ADFun_double::Range)
		.def("forward",   &ADFun_double::Forward)
		.def("reverse",   &ADFun_double::Reverse)
		.def("jacobian_", &ADFun_double::Jacobian)
	;
	// --------------------------------------------------------------------
	class_<AD_AD_double>("a2double", init<AD_double>())
		.def(str(self))
		PYCPPAD_OPERATOR_LIST
		PYCPPAD_STD_MATH_LIST(AD_double)
	;
	class_<ADFun_AD_double>("adfun_a_double", init< array& , array& >())
		.def("domain",    &ADFun_AD_double::Domain)
		.def("range",     &ADFun_AD_double::Range)
		.def("forward",   &ADFun_AD_double::Forward)
		.def("reverse",   &ADFun_AD_double::Reverse)
		.def("jacobian_", &ADFun_AD_double::Jacobian)
	;
}
