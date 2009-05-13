# $begin get_started.py$$ $newlinech #$$
# $spell
#	pycppad
# $$
#
# $section get_started: Example and Test$$
# $index get_started, example$$
# $index example, get_started$$
#
# For our getting started example, we consider the Gaussian density 
# for two independent random variables $latex F : \B{R}^2 \rightarrow \B{R}$$
# and its partial derivatives :
# $latex \[
# \begin{array}{rcl}
# F(x)                 & = & \exp \left[ - ( x_0^2  + x_1^2 ) / 2. \right] \\
# \partial_{x(0)} F(x) & = & - F(x) * x_0  \\
# \partial_{x(1)} F(x) & = & - F(x) * x_1
# \end{array}
# \] $$
# The following Python code computes these derivatives using $code pycppad$$
# and then checks the results for correctness: 
# $code
# $verbatim%example/get_started.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
def pycppad_test_get_started() :
  def F(x) :                                   # function to be differentiated
    return exp(-(x[0]**2. + x[1]**2.) / 2.)    # is Gaussian density
  x     = numpy.array( [ 1.,  2.] )
  a_x   = independent(x)
  a_y   = numpy.array( [ F(a_x) ] ) 
  f     = adfun(a_x, a_y)
  J     = f.jacobian(x)                        # J = F'(x)
  assert abs( J[0, 0] + F(x) * x[0] ) < 1e-10  # J[0,0] ~= - F(x) * x[0]
  assert abs( J[0, 1] + F(x) * x[1] ) < 1e-10  # J[0,1] ~= - F(x) * x[1]
# END CODE
