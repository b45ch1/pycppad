# $begin get_started.py$$ $newlinech #$$
#
# $section get_started: Example and Test$$
#
# $latex \[
# \begin{array}{rcl}
# F(x)                 & = & \exp \left[  ( x_0^2  + x_1^2 ) / 2. \right] \\
# \partial_{x(0)} F(x) & = & F(x) * x_0  \\
# \partial_{x(1)} F(x) & = & F(x) * x_1
# \end{array}
# \] $$
#
# $index get_started, example$$
# $index example, get_started$$
#
# $code
# $verbatim%example/get_started.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
def pycppad_test_get_started() :
  def F(x) :
    return exp( ( x[0] * x[0] + x[1] * x[1] ) / 2. )
  x     = numpy.array( [ 1.,  2.] )
  a_x   = independent(x)
  a_y   = numpy.array( [ F(a_x) ] )
  f     = adfun(a_x, a_y)
  J     = f.jacobian(x)
  delta = 10. * numpy.finfo(float).eps
  assert abs( J[0, 0] - F(x) * x[0] ) < delta 
  assert abs( J[0, 1] - F(x) * x[1] ) < delta 
# END CODE
