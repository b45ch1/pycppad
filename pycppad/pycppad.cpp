/*
---------------------------------------------------------------------------
$begin ad_unary$$
$spell
	numpy
$$

$section Unary Plus and Minus Operators$$

$index AD, unary operator$$
$index unary, AD operator$$
$index operator, AD unary$$
$index +, unary$$
$index -, unary$$
$index plus, unary$$
$index minus, unary$$

$head Syntax$$
$icode%y% = + %x
%$$
$icode%y% = - %x
%$$

$head Purpose$$
The operator $code +$$ ( $code -$$ ) above results
in $icode z$$ equal to $icode x$$ (minus $icode x$$). 

$head Type$$
The argument $icode x$$ can be $code a_float$$ or $code a2float$$
and the result $icode z$$ will have the same type as $icode x$$.

$head Arrays$$
The argument $icode x$$ may be
a $code numpy.array$$ with elements of type
$code a_float$$ or $code a2float$$.
In this case, the result $icode z$$ is an array with the same shape
and element type as $icode x$$.

$children%
	example/ad_unary.py%
	example/div_op.py
%$$
$head Example$$
The file $cref ad_unary.py$$ 
contains an example and test of these functions.

$end
---------------------------------------------------------------------------
$begin ad_numeric$$
$spell
	numpy
	op
$$

$section Binary Numeric Operators With an AD Result$$

$index AD, binary numeric operator$$
$index binary, AD numeric operator$$
$index operator, AD numeric binary$$
$index numeric, AD binary operator$$

$index +, binary$$
$index -, binary$$
$index *$$
$index /$$
$index **$$

$index plus, binary$$
$index minus, binary$$
$index times$$
$index divide$$
$index exponentiation$$ 

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
a $code numpy.array$$ with elements
that match one of possible type choices above.
If both $icode x$$ and $icode y$$ are arrays, they must have the same shape.
When either $icode x$$ or $icode y$$ is an array,
the result $icode z$$ is an array with the same shape.
The type of the elements of $icode z$$ correspond to the table above
(when the result type is a $code float$$,
this only refers to the element types matching as instances).

$children%
	example/ad_numeric.py
%$$
$head Example$$
The file $cref ad_numeric.py$$ 
contains an example and test of $code abs$$.

$end
---------------------------------------------------------------------------
$begin assign_op$$
$spell
	numpy
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
a $code numpy.array$$ with elements
that match one of possible type choices above.
If both $icode x$$ and $icode y$$ are arrays, they must have the same shape.
When either $icode x$$ or $icode y$$ is an array,
the result $icode z$$ is an array with the same shape.
The type of the elements of $icode z$$ correspond to the table above
(when the result type is a $code float$$,
this only refers to the element types matching as instances).

$children%
	example/assign_op.py
%$$
$head Example$$
The file $cref assign_op.py$$  
contains an example and test of these operations.

$end
---------------------------------------------------------------------------
$begin compare_op$$
$spell
	numpy
	yes yes
	bool
	op
$$

$section Binary Comparison Operators$$

$index bool, binary operator$$
$index binary, bool operator$$
$index operator, bool binary$$
$index comparison, binary operator$$
$index binary, comparison operator$$

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
$code ==$$ $cnext equal $rnext
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
a $code numpy.array$$ with elements
that match one of possible type choices above.
If both $icode x$$ and $icode y$$ are arrays, they must have the same shape.
When either $icode x$$ or $icode y$$ is an array,
the result $icode z$$ is an array with the same shape.
The type of the elements of $icode z$$ correspond to the table above
(when the result type is a $code float$$,
this only refers to the element types matching as instances).

$children%
	example/compare_op.py
%$$
$head Example$$
The file $cref compare_op.py$$  
contains an example and test of these operations.

$end
---------------------------------------------------------------------------
$begin std_math$$ 
$spell
	numpy
	arccos
	arcsin
	arctan
	cos
	exp
	tanh
	sqrt
$$

$section Standard Math Unary Functions$$

$index arccos$$
$index arcsin$$
$index arctan$$
$index cos$$
$index cosh$$
$index exp$$
$index log$$
$index log10$$
$index sin$$
$index sinh$$
$index sqrt$$
$index tan$$ 
$index tanh$$

$head Syntax$$
$icode%y% = %fun%(%x%)%$$

$head Purpose$$
Evaluate the standard math function $icode fun$$ where $icode fun$$
has one argument.

$head x$$
The argument $icode x$$ can be an instance of $code float$$,
an $code a_float$$, an $code a2float$$, or a $code numpy.array$$
of such objects.

$head y$$
If $icode x$$ is an instance of $code float$$,
$icode y$$ will also be an instance of $code float$$.
Otherwise $icode y$$ will have the same type as $icode x$$.
$pre

$$
In the case where $icode x$$ is an array, $icode y$$ will 
the same shape as $icode x$$ and the elements of $icode y$$
will have the  same type as the elements of $icode x$$.

$head fun$$
The function $icode fun$$ can be any of the following:
$code arccos$$,
$code arcsin$$,
$code arctan$$,
$code cos$$,
$code cosh$$,
$code exp$$,
$code log$$,
$code log10$$,
$code sin$$,
$code sinh$$,
$code sqrt$$,
$code tan$$, or
$code tanh$$.

$children%
	example/std_math.py
%$$
$head Example$$
The file $cref std_math.py$$ 
contains an example and test of these functions.

$end
---------------------------------------------------------------------------
$begin abs$$
$spell
	pycppad
	numpy
$$

$section Absolute Value Functions$$
$index abs$$

$head Syntax$$
$icode%y% = abs(%x%)%$$

$head Purpose$$
Sets $icode y$$ equal to the absolute value of $latex x$$.

$head x$$
The argument $icode x$$ can be an instance of $code float$$,
an $code a_float$$, an $code a2float$$, or an $code numpy.array$$ 
of such objects.

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
pycppad defines the derivative of the absolute value function by
$latex \[
	\R{abs}^{(1)} (x) = \R{sign} (x) = \left\{ \begin{array}{ll} 
		1 & \R{if} \; x > 0
		\\
		0 & \R{if} \; x = 0
		\\
		-1 & \R{if} \; x < 0
	\end{array} \right.
\] $$

$head Directional Derivative$$
Prior to 
$href%http://www.coin-or.org/CppAD/Doc/whats_new_11.htm#12-30%2011-12-30%$$,
$cref forward$$ mode computed the directional derivative
of the absolute value function which is defined by
$latex \[
	\R{abs}^\circ ( x , d ) = \lim_{\lambda \downarrow 0 } 
		\frac{\R{abs}(x + \lambda d) - \R{abs}(x) }{ \lambda }
\] $$ 
For $latex x \neq 0$$,
$latex \[
	\R{abs}^\circ ( x , d ) = \R{abs}^{(1)} ( x ) * d
\] $$
and $latex \R{abs}^\circ (0 , d) = |d|$$.

$children%
	example/abs.py
%$$
$head Example$$
The file $cref abs.py$$ 
contains an example and test of this function.

$end
---------------------------------------------------------------------------
$begin condexp$$
$spell
	condexp
	op
	lt
	le
	eq
	ge
	gt
	rel
$$

$section Conditional Expressions$$
$index abs$$

$head Syntax$$
$icode%result% = condexp_%rel%(%left%, %right%, %if_true%, %if_false%)%$$

$head Purpose$$
Record, as part of an operation sequence, the conditional result
$codei%
	if( %left% %op% %right% )
		%result% = %if_true%
	else	%result% = %if_false%
%$$
The relation $icode rel%$$, and operator $icode op$$,
have the following correspondence:
$codei%
	%rel%   lt   le   eq   ge   gt
	 %op%    <   <=   ==    >   >=
%$$

$head rel$$
In the syntax above, the relation $icode rel$$ represents one of the following
two characters: $code lt$$, $code le$$, $code eq$$, $code ge$$, $code gt$$. 
As in the table above,
$icode rel$$ determines which comparison operator $icode op$$ is used
when comparing $icode left$$ and $icode right$$.

$head left$$
The argument $icode left$$ must have type $code a_float$$ or $code a2float$$.
It specifies the value for the left side of the comparison operator.
 
$head right$$
The argument $icode right$$ must have the same type as $icode left$$.
It specifies the value for the right side of the comparison operator.

$head if_true$$
The argument $icode if_true$$ must have the same type as $icode left$$.
It specifies the return value if the result of the comparison is true.

$head if_false$$
The argument $icode if_false$$ must have the same type as $icode left$$.
It specifies the return value if the result of the comparison is false.

$head result$$
This result has the same type as $icode left$$.

$children%
	example/condexp.py
%$$
$head Example$$
The file $cref condexp.py$$ 
contains an example and test of these functions.
$end
---------------------------------------------------------------------------
*/
# include "environment.hpp"
# include "vector.hpp"
# include "vec2array.hpp"
# include "adfun.hpp"

# define PY_ARRAY_UNIQUE_SYMBOL PyArray_Pycppad

CppAD::AD<double> *truediv1(const CppAD::AD<double> &lhs, const CppAD::AD<double> &rhs){ return new CppAD::AD<double>(operator/(lhs,rhs));}
CppAD::AD<double> *truediv2(const CppAD::AD<double> &lhs, const double &rhs){ return new CppAD::AD<double>(operator/(lhs,rhs));}
CppAD::AD<double> *truediv3(const CppAD::AD<double> &rhs, const double &lhs){ return new CppAD::AD<double>(operator/(lhs,rhs));}


# define PYCPPAD_BINARY(op)       \
     .def(self     op self)       \
     .def(double() op self)       \
     .def(self     op double())


# define PYCPPAD_OPERATOR_LIST \
     .def(- self)              \
     .def(+ self)              \
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
          CppAD::Name; 

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

# define PYCPPAD_COND_EXP_LINK_CPP(Base) \
     CppAD::AD<Base> (* condexp_lt_AD_##Base ) (                  \
		const CppAD::AD<Base> &left     ,                       \
		const CppAD::AD<Base> &right    ,                       \
		const CppAD::AD<Base> &if_true  ,                       \
		const CppAD::AD<Base> &if_false ) = CppAD::CondExpLt;   \
     CppAD::AD<Base> (* condexp_le_AD_##Base ) (                  \
		const CppAD::AD<Base> &left     ,                       \
		const CppAD::AD<Base> &right    ,                       \
		const CppAD::AD<Base> &if_true  ,                       \
		const CppAD::AD<Base> &if_false ) = CppAD::CondExpLe;   \
     CppAD::AD<Base> (* condexp_eq_AD_##Base ) (                  \
		const CppAD::AD<Base> &left     ,                       \
		const CppAD::AD<Base> &right    ,                       \
		const CppAD::AD<Base> &if_true  ,                       \
		const CppAD::AD<Base> &if_false ) = CppAD::CondExpEq;   \
     CppAD::AD<Base> (* condexp_ge_AD_##Base ) (                  \
		const CppAD::AD<Base> &left     ,                       \
		const CppAD::AD<Base> &right    ,                       \
		const CppAD::AD<Base> &if_true  ,                       \
		const CppAD::AD<Base> &if_false ) = CppAD::CondExpGe;   \
     CppAD::AD<Base> (* condexp_gt_AD_##Base ) (                  \
		const CppAD::AD<Base> &left     ,                       \
		const CppAD::AD<Base> &right    ,                       \
		const CppAD::AD<Base> &if_true  ,                       \
		const CppAD::AD<Base> &if_false ) = CppAD::CondExpGt;

# define PYCPPAD_COND_EXP_LINK_PY(Base) \
	def("condexp_lt", condexp_lt_AD_##Base); \
	def("condexp_le", condexp_le_AD_##Base); \
	def("condexp_eq", condexp_eq_AD_##Base); \
	def("condexp_ge", condexp_ge_AD_##Base); \
	def("condexp_gt", condexp_gt_AD_##Base);


namespace pycppad {
	// Replacement for the CppAD error handler
	void cppad_error_handler(
		bool known           ,
		int  line            ,
		const char *file     ,
		const char *exp      ,
		const char *msg      )
	{	if( ! known ) msg = 
			"Bug detected in pycppad, Please report this.";

		// erorr handler must not return
		throw pycppad::exception(msg);
	}
	// This object lasts forever, so this is forever replacement of 
	// the default CppAD erorr handler 
	CppAD::ErrorHandler myhandler(cppad_error_handler);
	// call back function used by boost-python exception handler
	// to translate the exception to a user error message
	void translate_exception(pycppad::exception const& e)
	{	// Use the Python 'C' API to set up an exception object
		PyErr_SetString(PyExc_ValueError, e.what());
	}
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
	void abort_recording(void)
	{	AD_double::abort_recording();
		AD_AD_double::abort_recording();
		return;
	}
}
BOOST_PYTHON_MODULE(cppad_)
{
	// This tells boost-python which exception handler to use to use
	// when the throw in cppad_error_handler occurs
	boost::python::register_exception_translator<pycppad::exception>
		(&pycppad::translate_exception); 

	// AD_double is used in pycppad namespace
	typedef CppAD::AD<double>    AD_double;
	typedef CppAD::AD<AD_double> AD_AD_double;

	// standard math functions
	PYCPPAD_STD_MATH_LINK_CPP(double);
	PYCPPAD_STD_MATH_LINK_CPP(AD_double);
	// conditional expressions
	PYCPPAD_COND_EXP_LINK_CPP(double);
	PYCPPAD_COND_EXP_LINK_CPP(AD_double);

	// here are the things we are using from boost::python
	using boost::python::numeric::array;
	using boost::python::class_;
	using boost::python::init;
	using boost::python::self;
	using boost::python::self_ns::str;
	using boost::python::def;
    using boost::python::manage_new_object;
    using boost::python::return_value_policy;

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
	// documented in adfun.py
	def("abort_recording", pycppad::abort_recording);
	// conditional expressions
	PYCPPAD_COND_EXP_LINK_PY(double)
	PYCPPAD_COND_EXP_LINK_PY(AD_double)
	// --------------------------------------------------------------------
	class_<AD_double>("a_float", init<double>())
		.def(str(self))
		PYCPPAD_OPERATOR_LIST
		PYCPPAD_STD_MATH_LINK_PY(double)

		// abs
		.def( abs(self) )

		// pow
     	.def( pow(self, self) ) 
     	.def( pow(self, double()) )
     	.def( pow(self, int()) )
     	.def( pow(double(), self) )

        // truediv (returns python 3 division even when in python 2)
        .def("__truediv__", truediv1, return_value_policy<manage_new_object>())
        .def("__truediv__", truediv2, return_value_policy<manage_new_object>())
        .def("__rtruediv__", truediv3, return_value_policy<manage_new_object>())

	;

	class_<ADFun_double>("adfun_float", init< array& , array& >())
		.def("domain",    &ADFun_double::Domain)
		.def("forward",   &ADFun_double::Forward)
		.def("compare_change",   &ADFun_double::CompareChange)
		.def("hessian_" , &ADFun_double::Hessian)
		.def("jacobian_", &ADFun_double::Jacobian)
		.def("optimize",  &ADFun_double::optimize)
		.def("range",     &ADFun_double::Range)
		.def("reverse",   &ADFun_double::Reverse)
	;
	// --------------------------------------------------------------------
	class_<AD_AD_double>("a2float", init<AD_double>())
		.def(str(self))
		PYCPPAD_OPERATOR_LIST
		PYCPPAD_STD_MATH_LINK_PY(AD_double)

		// abs
		.def( abs(self) )

		// pow
     	.def( pow(self, self) ) 
     	.def( pow(self, double()) )
     	.def( pow(double(), self) )
	;
	class_<ADFun_AD_double>("adfun_a_float", init< array& , array& >())
		.def("domain",    &ADFun_AD_double::Domain)
		.def("range",     &ADFun_AD_double::Range)
		.def("forward",   &ADFun_AD_double::Forward)
		.def("compare_change",   &ADFun_AD_double::CompareChange)
		.def("reverse",   &ADFun_AD_double::Reverse)
		.def("jacobian_", &ADFun_AD_double::Jacobian)
		.def("hessian_",  &ADFun_AD_double::Hessian)
	;
}

