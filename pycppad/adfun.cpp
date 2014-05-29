/*
---------------------------------------------------------------------------
$begin forward$$
$spell
	numpy
	adfun
	Taylor
$$

$section  Forward Mode: Derivative in One Domain Direction$$

$index forward$$
$index domain, direction derivative$$
$index derivative, domain direction$$
$index direction, domain derivative$$


$head Syntax$$
$icode%y_p% = %f%.forward(%p%, %x_p%)%$$

$head Purpose$$
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ to denote the 
function corresponding to the $code adfun$$ object $cref/f/adfun/f/$$.
Given the $th p$$ order Taylor expansion for a function
$latex X : \B{R} \rightarrow \B{R}^n$$, this function can be used
to compute the $th p$$ order Taylor expansion for the function
$latex Y : \B{R} \rightarrow \B{R}^m$$ defined by
$latex \[
	Y(t) = F [ X(t) ]
\] $$

$head x_k$$
For $latex k = 0 , \ldots , p$$,
we use $latex x^{(k)}$$ to denote the value of $icode x_k$$ in the
most recent call to
$codei%
	%f%.forward(%k%, %x_k%)
%$$
including $latex x^{(p)}$$ as the value $icode x_p$$ in this call.
We define the function $latex X(t)$$ by
$latex \[
	X(t) =  x^{(0)} + x^{(1)} * t + \cdots + x^{(p)} * t^p 
\] $$

$head y_k$$
For $latex k = 0 , \ldots , p$$,
we use $latex y^{(k)}$$ to denote the Taylor coefficients
for $latex Y(t) = F[ X(t) ]$$ expanded about zero; i.e.,
$latex \[
\begin{array}{rcl}
y^{(k)} & = & Y^{(k)} (0) / k !
\\
Y(t)    & = & y^{(0)} + y^{(1)} * t + \cdots + y^{(p)} * t^p + o( t^p )
\end{array}
\] $$
where $latex o( t^p ) / t^p \rightarrow 0$$ as $latex t \rightarrow 0$$.
The coefficient $latex y^{(p)}$$ is equal to 
the value $icode y_p$$ returned by this call.

$head f$$
The object $icode f$$ must be an $cref adfun$$ object.
We use $cref/level/adfun/f/level/$$ for the AD $cref ad$$ level of 
this object.

$head p$$
The argument $icode p$$ is a non-negative $code int$$.
It specifies the order of the Taylor coefficient for $latex Y(t)$$
that is computed.

$head x_p$$
The argument $icode x_p$$ is a $code numpy.array$$ with one dimension
(i.e., a vector) with length equal to the domain size $cref/n/adfun/f/n/$$
for the function $icode f$$.
It specifies the $th p$$ order Taylor coefficient for $latex X(t)$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode x_p$$ must be either $code int$$ or instances
of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode x_p$$ must be $code a_float$$ objects.


$head y_p$$
The return value $icode y_p$$ is a $code numpy.array$$ with one dimension
(i.e., a vector) with length equal to the range size $cref/m/adfun/f/m/$$
for the function $icode f$$.
It is set to the $th p$$ order Taylor coefficient for $latex Y(t)$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode y_p$$ will be instances of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode y_p$$ will be $code a_float$$ objects.

$children%
	example/forward_0.py%
	example/forward_1.py
%$$
$head Example$$
$table
$rref forward_0.py$$
$rref forward_1.py$$
$tend

$end
---------------------------------------------------------------------------
$begin reverse$$
$spell
	dw
	numpy
	adfun
	Taylor
$$

$section  Reverse Mode: Derivative in One Range Direction$$

$index reverse$$
$index derivative, range direction$$
$index range, direction derivative$$
$index direction, range derivative$$

$head Syntax$$
$icode%dw% = %f%.forward(%p%, %w%)%$$

$head Purpose$$
Reverse mode computes the derivative of the $cref forward$$ more 
Taylor coefficients with respect to the domain variable $latex x$$.

$head x_k$$
For $latex k = 0 , \ldots , p$$,
we use $latex x^{(k)}$$ to denote the value of $icode x_k$$ in the
most recent call to
$codei%
	%f%.forward(%k%, %x_k%)
%$$
We use $latex F : \B{R}^n \rightarrow \B{R}^m$$ to denote the 
function corresponding to the $code adfun$$ object $cref/f/adfun/f/$$.

$head X(t, u)$$
We define the function $latex X : \B{R} \times \B{R}^n \rightarrow \B{R}^n$$ by
$latex \[
	X(t, u) =  u + x^{(0)} + x^{(1)} * t + \cdots + x^{(p-1)} * t^{p-1} 
\] $$
Note that for $latex k = 0 , \ldots , p - 1$$,
$latex \[
	x^{(k)} = \frac{1}{k !} \frac{\partial^k}{\partial t^k} X(0, 0)
\] $$

$head W(t, u)$$
The function $latex W : \B{R} \times \B{R}^n \rightarrow \B{R}$$
is defined by
$latex \[
W(t, u) = w_0 * F_0 [ X(t, u) ] + \cdots + w_{m-1} * F_{m-1} [ X(t, u) ]
\] $$
We define the function $latex W_k : \B{R}^n \rightarrow \B{R}$$ by
$latex \[
	W_k ( u ) = \frac{1}{k !} \frac{\partial^k}{\partial t^k} W(0, u)
\] $$
It follows that
$latex \[
W(t, u ) = W_0 ( u ) + W_1 ( u ) * t + \cdots + W_{p-1} (u) * t^{p-1}
         + o( t^{p-1} )
\] $$
where $latex o( t^{p-1} ) / t^{p-1} \rightarrow 0$$
as $latex t \rightarrow 0$$.

$head f$$
The object $icode f$$ must be an $cref adfun$$ object.
We use $cref/level/adfun/f/level/$$ for the AD $cref ad$$ level of 
this object.

$head p$$
The argument $icode p$$ is a non-negative $code int$$.
It specifies the order of the Taylor coefficient $latex W_{p-1} ( u )$$
that is differentiated.
Note that $latex W_{p-1} (u)$$ corresponds a derivative of order 
$latex p-1$$ of $latex F(x)$$,
so the derivative of $latex W_{p-1} (u)$$ corresponds to a derivative
of order $latex p$$ of $latex F(x)$$.

$head w$$
The argument $icode w$$ is a $code numpy.array$$ with one dimension
(i.e., a vector) with length equal to the range size $cref/m/adfun/f/m/$$
for the function $icode f$$.
It specifies the weighting vector $latex w$$ used in the definition of
$latex W(t, u)$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode w$$ must be either $code int$$ or instances
of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode w$$ must be $code a_float$$ objects.

$head dw$$
The return value $icode v$$ is a $code numpy.array$$ with one dimension
(i.e., a vector) with length equal to the domain size $cref/n/adfun/f/n/$$
for the function $icode f$$.
It is set to the derivative 
$latex \[
\begin{array}{rcl}
dw & = & W_{p-1}^{(1)} ( 0 ) \\
& = &
\partial_u \frac{1}{(p-1) !} \frac{\partial^{p-1}}{\partial t^{p-1}} W(0, 0)
\end{array}
\] $$
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode dw$$ will be instances of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode dw$$ will be $code a_float$$ objects.

$head First Order$$
In the case where $latex p = 1$$, we have
$latex \[
\begin{array}{rcl}
dw 
& = & \partial_u \frac{1}{0 !} \frac{\partial^0}{\partial t^0} W(0, 0)
\\
& = & \partial_u W(0, 0)
\\
& = & 
\partial_u \left[ 
w_0 * F_0 ( u + x^{(0)} ) + \cdots +  w_{m-1} F_{m-1} ( u + x^{(0)} )
\right]_{u = 0}
\\
& = & 
w_0 * F_0^{(1)} ( x^{(0)} ) + \cdots + w_{m-1} * F_{m-1}^{(1)} ( x^{(0)} )
\end{array}
\] $$

$head Second Order$$
In the case where $latex p = 2$$, we have
$latex \[
\begin{array}{rcl}
dw 
& = & \partial_u \frac{1}{1 !} \frac{\partial^1}{\partial t^1} W (0, 0)
\\
& = &
\partial_u \left[
	w_0 * F_0^{(1)} ( u + x^{(0)} ) * x^{(1)}
 	+ \cdots + 
	w_{m-1} * F_{m-1}^{(1)} ( u + x^{(0)} ) * x^{(1)}
\right]_{u = 0}
\\
& = &
w_0 * ( x^{(1)} )^\R{T} * F_0^{(2)} ( x^{(0)} ) 
+ \cdots + 
w_{m-1} * ( x^{(1)} )^\R{T} * F_{m-1}^{(2)} ( x^{(0)} )
\end{array}
\] $$

$children%
	example/reverse_1.py%
	example/reverse_2.py
%$$
$head Example$$ 
$table
$rref reverse_1.py$$
$rref reverse_2.py$$
$tend

$end
---------------------------------------------------------------------------
$begin jacobian$$
$spell
	jacobian
	numpy
	adfun
$$

$section Driver for Computing Entire Derivative$$

$index jacobian$$
$index driver, entire derivative$$
$index entire, derivative driver$$
$index derivative, entire driver$$

$head Syntax$$
$icode%J% = %f%.jacobian(%x%)%$$

$head Purpose$$
This routine computes the entire derivative $latex F^{(1)} (x)$$
where $latex F : \B{R}^n \rightarrow \B{R}^m$$ is the 
function corresponding to the $code adfun$$ object $cref/f/adfun/f/$$.

$head f$$
The object $icode f$$ must be an $cref adfun$$ object.
We use $cref/level/adfun/f/level/$$ for the AD $cref ad$$ level of 
this object.

$head x$$
The argument $icode x$$ is a $code numpy.array$$ with one dimension
(i.e., a vector) with length equal to the domain size $cref/n/adfun/f/n/$$
for the function $icode f$$.
It specifies the argument value at which the derivative is computed.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode x$$ must be either $code int$$ or instances
of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode x$$ must be $code a_float$$ objects.

$head J$$
The return value $icode J$$ is a $code numpy.array$$ with two dimensions
(i.e., a matrix).
The first dimension (row size) is equal to $cref/m/adfun/f/m/$$
(the number of range components in the function $icode f$$).
The second dimension (column size) is equal to $cref/n/adfun/f/n/$$
(the number of domain components in the function $icode f$$).
It is set to the derivative; i.e.,
$latex \[
	J = F^{(1)} (x)
\] $$
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode J$$ will be instances of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode J$$ will be $code a_float$$ objects.

$children%
	example/jacobian.py
%$$
$head Example$$ 
The file $cref jacobian.py$$ contains an example and test of this operation.

$end
---------------------------------------------------------------------------
$begin hessian$$
$spell
	hessian
	numpy
	adfun
$$

$section Driver for Computing Hessian in a Range Direction$$

$index hessian, driver$$
$index driver, hessian$$
$index Lagrangian, hessian$$
$index hessian, Lagrangian$$

$head Syntax$$
$icode%H% = %f%.hessian(%x%, %w%)%$$

$head Purpose$$
This routine computes the Hessian of the weighted sum
$latex \[
	w_0 * F_0 (x) + \cdots + w_{m-1} * F_{m-1} (x)
\] $$
where $latex F : \B{R}^n \rightarrow \B{R}^m$$ is the 
function corresponding to the $code adfun$$ object $cref/f/adfun/f/$$.

$head f$$
The object $icode f$$ must be an $cref adfun$$ object.
We use $cref/level/adfun/f/level/$$ for the AD $cref ad$$ level of 
this object.

$head x$$
The argument $icode x$$ is a $code numpy.array$$ with one dimension
(i.e., a vector) with length equal to the domain size $cref/n/adfun/f/n/$$
for the function $icode f$$.
It specifies the argument value at which the derivative is computed.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode x$$ must be either $code int$$ or instances
of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode x$$ must be $code a_float$$ objects.

$head w$$
The argument $icode w$$ is a $code numpy.array$$ with one dimension
(i.e., a vector) with length equal to the range size $cref/m/adfun/f/m/$$
for the function $icode f$$.
It specifies the argument value at which the derivative is computed.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode w$$ must be either $code int$$ or instances
of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode w$$ must be $code a_float$$ objects.

$head H$$
The return value $icode H$$ is a $code numpy.array$$ with two dimensions
(i.e., a matrix).
Both its first and second dimension size 
(row and column size) are equal to $cref/n/adfun/f/n/$$
(the number of domain components in the function $icode f$$).
It is set to the Hessian; i.e.,
$latex \[
	H = w_0 * F_0^{(2)} (x) + \cdots + w_{m-1} * F_{m-1}^{(2)} (x)
\] $$
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is zero,
all the elements of $icode H$$ will be instances of $code float$$.
If the AD $cref/level/adfun/f/level/$$ for $icode f$$ is one,
all the elements of $icode H$$ will be $code a_float$$ objects.

$children%
	example/hessian.py
%$$
$head Example$$ 
The file $cref hessian.py$$ contains an example and test of this operation.

$end
---------------------------------------------------------------------------
$begin optimize$$
$spell
	Taylor
	var
$$

$section Optimize an AD Function Object Tape$$

$index optimize$$
$index tape, optimize$$
$index sequence, optimize operations$$
$index operations, optimize sequence$$
$index speed, optimize$$
$index memory, optimize$$

$head Syntax$$
$icode%f%.optimize()%$$


$head Purpose$$
The operation sequence corresponding to an $cref ADFun$$ object can
be very large and involve many operations.
The $icode%f%.optimize%$$ procedure reduces the number of operations,
and thereby the time and memory, required to
compute function and derivative values. 

$head f$$
The object $icode f$$ is an $cref adfun$$ object.

$head Efficiency$$
The $code optimize$$ member function
may greatly reduce the size of the operation sequence corresponding to 
$icode f$$.

$children%
	example/optimize.py
%$$
$head Example$$ 
The file $cref optimize.py$$ contains an example and test of this operation.

$end
---------------------------------------------------------------------------
*/
# include "adfun.hpp"
# include "vector.hpp"
# include "vec2array.hpp"

namespace pycppad {
	// -------------------------------------------------------------

	// constructor for python class ADFun<Base>
	template <class Base>
	ADFun<Base>::ADFun(array& x_array, array& y_array)
	{	vec< CppAD::AD<Base> > x_vec(x_array);
		vec< CppAD::AD<Base> > y_vec(y_array);

		f_.Dependent(x_vec, y_vec);
	}

	// Domain
	template <class Base>
	int ADFun<Base>::Domain(void)
	{	return static_cast<int>( f_.Domain() ); }

	// Range
	template <class Base>
	int ADFun<Base>::Range(void)
	{	return static_cast<int>( f_.Range() ); }

	// Forward
	template <class Base>
	array ADFun<Base>::Forward(int p, array& xp)
	{	size_t    p_sz(p);
		vec<Base> xp_vec(xp);
		vec<Base> result = f_.Forward(p_sz, xp_vec);
		return vec2array(result);
	}

# ifdef NDEBUG
	template <class Base>
	int ADFun<Base>::CompareChange(void)
	{	bool known = true;
		CppAD::ErrorHandler::Call(
			known,
			__LINE__,
			__FILE__,
			"# ifndef NDEBUG",
			"Cannot use CompareChange when NDEBUG is defined"
		);
		return 0;
	}
# else
	// CompareChange
	template <class Base>
	int ADFun<Base>::CompareChange(void)
	{	return static_cast<int>( f_.CompareChange() ); }
# endif

	// Reverse
	template <class Base>
	array ADFun<Base>::Reverse(int p, array& w)
	{	size_t    p_sz(p);
		vec<Base> w_vec(w);
		vec<Base> dw_vec = f_.Reverse(p_sz, w_vec);
		size_t n = f_.Domain();
		vec<Base> result(n);
		for(size_t j = 0; j < n; j++)
			result[j] = dw_vec[j*p + p - 1];
		return vec2array(result);
	}

	// Jacobian
	template <class Base>
	array ADFun<Base>::Jacobian(array& x)
	{	vec<Base> x_vec(x);
		vec<Base> result = f_.Jacobian(x_vec);
		// Kludge: return a vector which is reshaped by cppad.py
		return vec2array(result);
	}

	// Hessian
	template <class Base>
	array ADFun<Base>::Hessian(array& x, array& w)
	{	vec<Base> x_vec(x);
		vec<Base> w_vec(w);
		vec<Base> result = f_.Hessian(x_vec, w_vec);
		// Kludge: return a vector which is reshaped by cppad.py
		return vec2array(result);
	}

	// optimize
	template <class Base>
	void ADFun<Base>::optimize(void)
	{	f_.optimize(); }

	// -------------------------------------------------------------
	// instantiate instances of ADFun<Base>
	template class ADFun<double>;
	template class ADFun<AD_double>;
	// -------------------------------------------------------------

	void adfun_avoid_warning_that_import_array_not_used(void)
	{	import_array(); }
}
