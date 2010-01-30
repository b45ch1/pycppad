# $begin runge_kutta_4_cpp.py$$ $newlinech #$$
# $spell
#	runge_kutta
#	numpy
#	pycppad
#	def
#	dt
#	yi
#	yf
# $$
# $index example, runge_kutta_4$$
# $index C++ speed, python function$$
# $index speed, python function$$
# $index tape, python function$$
# $index runge_kutta_4, evaluate solution$$
#
# $section runge_kutta_4 With C++ Speed: Example and Test$$
#
# $head Purpose$$
# In this example we should how any Python function can be recorded
# in a $code pycppad$$ function object and then evaluated at much
# higher speeds than the Python evaluation.
#
# $head Discussion$$
# Define $latex y : \B{R}^2 \times \B{R} \rightarrow \B{R}^n$$ by
# $latex \[
#	\begin{array}{rcl}
#	y(x, 0)            & = & x_0 
#	\\
#	\partial_t y(x, t) & = & x_1 y(x, t)
#	\end{array}
# \] $$ 
# It follows that 
# $latex \[
#	y(x, t) = x_0 \exp ( x_1 t )
# \] $$
# Suppose we want to compute values for the function 
# $latex g : \B{R}^2 \rightarrow \B{R} $$ defined by
# $latex \[
#	g(x) = y(x, 1)
# \] $$
# In this example we compare the execution time for doing this in pure Python
# with using a pycppad function object to compute $latex g(x)$$ in C++.  
#
# $head Source Code$$
#
# $code
# $verbatim%example/runge_kutta_4_cpp.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
import time
def pycppad_test_runge_kutta_4_cpp() :
	x_1 = 0;   # use this variable to switch x_1 between float and ad(float)
	def fun(t , y) :
		f     = x_1 * y
		return f
	# Number of Runge-Kutta times steps to include in the function object
	M = 10 

	# Start time for recording the pycppad function object
	s0  = time.time()
	# Declare three independent variables. The operation sequence does not
	# depend on x, so we could use any value here.
	x    = numpy.array( [.1, .1, .1] )
	a_x  = independent( numpy.array( x ) )
	# First independent variables, x[0], is the value of y(0)
	a_y = numpy.array( [ a_x[0] ] )
	# Make x_1 a variable so can use rk4 with various coefficients.
	x_1 = a_x[1]
	# Make dt a variable so can use rk4 with various step sizes.
	dt  = a_x[2]
	# f(t, y) does not depend on t, so no need to make t a variable.
	t   = ad(0.)
	# Record the operations for 10 time step
	for k in range(M) :
		a_y = runge_kutta_4(fun, t, a_y, dt)
		t   = t + dt
	# define the AD function rk4 : x -> y
	rk4 = adfun(a_x, a_y)
	# amount of time it took to tape this function object
	tape_sec =  time.time() - s0

	ti  = 0.              # initial time
	tf  = 1.              # final time
	N   = M * 100         # number of time steps 
	dt  = (tf - ti) / N   # size of time step
	x_0 = 2.              # use this for initial value of y(t)
	x_1 = .5              # use this for coefficient in ODE

	# python version of integrator with float values
	s0  = time.time()
	t   = ti
	y   = numpy.array( [ x_0 ] ); 
	for k in range(N) :
		y = runge_kutta_4(fun, t, y, dt)
		t = t + dt
	# number of seconds to solve the ODE using python float
	python_sec =  time.time() - s0
	# check solution is correct
	assert( abs( y[0] - x_0 * exp( x_1 * tf ) ) < 1e-10 ) 

	# pycppad function object version of integrator 
	s0  = time.time()
	t   = ti
	x   = numpy.array( [ x_0 , x_1 , dt ] ) 
	for k in range(N/M) :
		y    = rk4.forward(0, x);
		x[0] = y[0];
	# number of seconds to solve the ODE using python float
	cpp_sec =  time.time() - s0
	# check solution is correct
	assert( abs( y[0] - x_0 * exp( x_1 * tf ) ) < 1e-10 ) 
	
	# check that C++ is always more than 20 times faster
	assert( 20. * cpp_sec <= python_sec )

	# Actual factor is about 100. Uncomment the print statement below to 
	# see it for your machine / optimized or debug build.
	format = 'cpp_sec = %8f, python_sec/cpp_sec = %5.1f'
	format = format + ', tape_sec/cpp_sec = %5.1f'
	print format % ( cpp_sec, python_sec/cpp_sec, tape_sec/cpp_sec )

# END CODE
