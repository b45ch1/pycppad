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

$head Derivative$$
Define $latex F(x) = \R{abs}(x)$$. It follows that
$latex \[
	F^{(1)} (x) = \left\{ \begin{array}{ll} 
		1 & \R{if} \; x > 0
		\\
		-1 & \R{if} \; x < 0
	\end{array} \right.
\] $$
and the derivative $latex F^{(1)} (0)$$ does not exist.

$head Directional Derivative$$
On the other hand, for the absolute value function,
$cref/forward/$$ mode computes directional derivatives
which are defined by
$latex \[
	F^\circ ( x , d ) = \lim_{\lambda \downarrow 0 } 
		\frac{F(x + \lambda d) - F(x) }{ \lambda }
\] $$ 
For $latex x \neq 0$$,
$latex \[
	F^\circ ( x , d ) = F^{(1)} ( x ) * d
\] $$
and $latex F^\circ (0 , d) = |d|$$.

$children%
	example/abs.py
%$$
$head Example$$
The file $cref/abs.py/$$ 
contains an example and test of these functions.

$end
---------------------------------------------------------------------------
$begin ad_binary$$ $newlinech #$$
$spell
	op
$$

$section Binary Operators With an AD Result$$

$index AD, binary operator$$
$index binary, AD operator$$
$index operator, AD binary$$

$index +$$
$index -$$
$index *$$
$index /$$
$index **$$

$head Syntax$$
$icode%z% = %x% %op% %y%$$

$head Purpose$$
Sets $icode z$$ to the result of the binary operation defined by $icode op$$
and with $icode x$$ as the left operand and $icode y$$ as the right operand.

$head op$$
The possible values for $icode op$$ are
$table
$icode op$$ $pre  $$ $cnext Meaning $rnext
$code +$$  $cnext addition  $rnext
$code -$$  $cnext subtraction  $rnext
$code *$$  $cnext multiplication  $rnext
$code /$$  $cnext division  $rnext
$code **$$ $cnext exponentiation
$tend

$head Types$$
The following table lists the possible types for $icode x$$ and $icode y$$
and the corresponding result type for $icode z$$.
$codei%
                      %y%
 %x%           float    a_float   a2float
         %-------------------------------%
 float   %-%   float    a_float   a2float
a_float  %-%  a_float   a_float
a2float  %-%  a2float             a2float
%$$
The type $code float$$ does not need to be matched exactly
but rather as an instance of $code float$$.

$head Arrays$$
Either $icode x$$ or $icode y$$ or both may be
an $cref/array/$$ with elements
that match one of possible type choices above.
If both $icode x$$ and $icode y$$ are arrays, they must have the same shape.
When either $icode x$$ or $icode y$$ is an array,
the result $icode z$$ is an array with the same shape.
The type of the elements of $icode z$$ correspond to the table above
(when the result type is a $code float$$,
this only refers to the element types matching as instances).

$children%
	example/ad_binary.py%
	example/ad_binary_a2.py
%$$
$head Example$$
The file $cref/ad_binary.py/$$  ($cref/ad_binary_a2.py/$$)
contains an example and test of these operations using 
$code a_float$$ ($code a2float$$).

$end
---------------------------------------------------------------------------
$begin assign_op$$ $newlinech #$$
$spell
	op
$$

$section Computed Assignment Operators$$

$index computed, assignment operator$$
$index assignment, computed operator$$
$index operator, computed assignment$$

$index +=$$
$index -=$$
$index *=$$
$index /=$$

$head Syntax$$
$icode%u% %op%= %x%$$

$head Purpose$$
We use $icode y$$ ($icode z$$) to refer to the value of 
$icode u$$ before (after) the operation.
This operation sets $icode z$$ equal to 
$codei%
	%y% %op% %x%
%$$.

$head op$$
The possible values for $icode op$$ are
$table
$icode op$$ $pre  $$ $cnext Meaning $rnext
$code +$$  $cnext addition  $rnext
$code -$$  $cnext subtraction  $rnext
$code *$$  $cnext multiplication  $rnext
$code /$$  $cnext division 
$tend

$head Types$$
The following table lists the possible types for $icode x$$ and $icode y$$
(the value of $icode u$$ before the operation)
and the corresponding $icode z$$
(the value of $icode u$$ after the operation).
$codei%
                      %y%
 %x%           float    a_float   a2float
         %-------------------------------%
 float   %-%   float    a_float   a2float
a_float  %-%  a_float   a_float
a2float  %-%  a2float             a2float
%$$
The type $code float$$ does not need to be matched exactly
but rather as an instance of $code float$$.

$head Arrays$$
Either $icode x$$ or $icode y$$ or both may be
an $cref/array/$$ with elements
that match one of possible type choices above.
If both $icode x$$ and $icode y$$ are arrays, they must have the same shape.
When either $icode x$$ or $icode y$$ is an array,
the result $icode z$$ is an array with the same shape.
The type of the elements of $icode z$$ correspond to the table above
(when the result type is a $code float$$,
this only refers to the element types matching as instances).

$children%
	example/assign_op.py%
	example/assign_op_a2.py
%$$
$head Example$$
The file $cref/assign_op.py/$$  ($cref/assign_op_a2.py/$$)
contains an example and test of these operations using 
$code a_float$$ ($code a2float$$).

$end
---------------------------------------------------------------------------
$begin bool_binary$$ $newlinech #$$
$spell
	yes yes
	bool
	op
$$

$section Binary Operators With a Boolean Result$$

$index bool, binary operator$$
$index binary, bool operator$$
$index operator, bool binary$$

$index >$$
$index <$$
$index >=$$
$index <=$$
$index ==$$
$index !=$$

$head Syntax$$
$icode%z% = %x% %op% %y%$$

$head Purpose$$
Sets $icode z$$ to the result of the binary operation defined by $icode op$$
and with $icode x$$ as the left operand and $icode y$$ as the right operand.

$head op$$
The possible values for $icode op$$ are
$table
$icode op$$ $pre  $$ $cnext Meaning $rnext
$code >$$  $cnext greater than  $rnext
$code >$$  $cnext less than  $rnext
$code >=$$  $cnext greater than or equal  $rnext
$code <=$$  $cnext less than or equal  $rnext
$code ==$$ $cnext equal
$code !=$$ $cnext not equal
$tend

$head Types$$
The following table lists the possible (yes) and impossible (no)
types for $icode x$$ and $icode y$$.
The corresponding result type for $icode z$$ is always $code bool$$.
$codei%
                      %y%
 %x%           float    a_float   a2float
         %-------------------------------%
 float   %-%    yes      yes       yes   
a_float  %-%    yes      yes       no
a2float  %-%    yes      no        yes 
%$$
The type $code float$$ does not need to be matched exactly
but rather as an instance of $code float$$.

$head Arrays$$
Either $icode x$$ or $icode y$$ or both may be
an $cref/array/$$ with elements
that match one of possible type choices above.
If both $icode x$$ and $icode y$$ are arrays, they must have the same shape.
When either $icode x$$ or $icode y$$ is an array,
the result $icode z$$ is an array with the same shape.
The type of the elements of $icode z$$ correspond to the table above
(when the result type is a $code float$$,
this only refers to the element types matching as instances).

$children%
	example/bool_binary.py%
	example/bool_binary_a2.py
%$$
$head Example$$
The file $cref/bool_binary.py/$$  ($cref/bool_binary_a2.py/$$)
contains an example and test of these operations using 
$code a_float$$ ($code a2float$$).

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

	PYCPPAD_STD_MATH_LINK_CPP(AD_double);

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
     		.def( pow(self, self) ) 
     		.def( pow(self, double()) )
     		.def( pow(double(), self) )
		.def( abs(self) )
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
     		.def( pow(self, self) ) 
     		.def( pow(self, double()) )
     		.def( pow(double(), self) )
		.def( abs(self) )
	;
	class_<ADFun_AD_double>("adfun_a_float", init< array& , array& >())
		.def("domain",    &ADFun_AD_double::Domain)
		.def("range",     &ADFun_AD_double::Range)
		.def("forward",   &ADFun_AD_double::Forward)
		.def("reverse",   &ADFun_AD_double::Reverse)
		.def("jacobian_", &ADFun_AD_double::Jacobian)
	;
}
