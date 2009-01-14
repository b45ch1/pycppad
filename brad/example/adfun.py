# $begin adfun.py$$ $newlinech #$$
# $spell
#	adfun
# $$
#
# $section adfun: Example and Test$$
#
# $index adfun, example$$
# $index example, adfun$$
#
# $code
# $verbatim%example/adfun.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_adfun() :
  # record operations at x = (0, 0, 0)
  x    = numpy.array( [ 0., 0., 0. ] )
  a_x  = independent(x)   # declare independent variables and start recording
  a_y0 = a_x[0];
  a_y1 = a_x[0] * a_x[1];
  a_y2 = a_x[0] * a_x[1] * a_x[2];
  a_y  = numpy.array( [ a_y0, a_y1, a_y2 ] )
  f    = adfun(a_x, a_y)  # declare dependent variables and stop recording
  # evaluate function at x = (1, 2, 3)
  x    = numpy.array( [ 1., 2., 3. ] ) 
  y    = f.forward(0, x) 
  assert y[0] == x[0]
  assert y[1] == x[0] * x[1]
  assert y[2] == x[0] * x[1] * x[2]
# END CODE
