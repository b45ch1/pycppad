# $begin runge_kutta_4_ad.py$$ $newlinech #$$
# $spell
#	runge_kutta
#	numpy
#	pycppad
#	def
#	dt
#	yi
#	yf
# $$
#
# $section runge_kutta_4 An AD Example and Test$$
#
# $head Discussion$$
# Define $latex y : \B{R} \times \B{R} \rightarrow \B{R}^n$$ by
# $latex \[
#	y_j (x, t) =  x t^{j+1}
# \] $$ 
# It follows that the derivative of $latex y(t)$$ satisfies the 
# $cref/runge_kutta_4/$$ ODE equation where 
# $latex y(0) = 0 $$ and $latex f(t, y)$$ is given by
# $latex \[
# f(t , y)_j = \partial_t y_j (x, t) = \left\{ \begin{array}{ll}
#  x                    & {\; \rm if \;} j = 0      \\
#  (j+1) y_{j-1} (x, t) & {\; \rm otherwise }
# \end{array} \right.
# \] $$
# It also follows that
# $latex \[
#	\partial_x y_j (x, t) = t^{j+1}
# \] $$
#
# $head Source Code$$
#
# $code
# $verbatim%example/runge_kutta_4_ad.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
def pycppad_test_runge_kutta_4_ad() :
	def f(t , y) :
		n        = y.size
		f        = ad( numpy.zeros(n) )
		f[0]     = a_x[0]
		index    = numpy.array( range(n-1) ) + 1
		f[index] = (index + 1) * y[index-1] 
		return f
	n  = 5         # size of y(t) (order of method plus 1)
	ti = 0.        # initial time
	dt = 2.        # a very large time step size (method is exact)

	# initial value for y(t); i.e., y(0)
	a_yi = ad( numpy.zeros(n) ) 
	
	# declare a_x to be the independent variable vector
	x    = numpy.array( [.5] )
	a_x  = independent( numpy.array( x ) )

	# take one 4-th order Runge-Kutta integration step of size dt 
	a_yf = runge_kutta_4(f, ti, a_yi, dt)

	# define the AD function g : x -> yf
	g = adfun(a_x, a_yf)

	# compute the derivative of g w.r.t x at x equals .5
	dg = g.jacobian(x)
	
	# check the result is as expected
	t_jp = 1                                        # t^0
	for j in range(n-1) :
		t_jp  = t_jp * dt                          # t^(j+1) at t = dt
		assert abs( a_yf[j] - x[0]*t_jp ) < 1e-10  # check yf[j] = x*t^(j+1_
		assert abs( dg[j,0] - t_jp    ) < 1e-10    # check dg[j] = t^(j+1)
# END CODE
