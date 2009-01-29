# $begin jacobian.py$$ $newlinech #$$
# $spell
#	Jacobian
# $$
#
# $section Entire Derivative: Example and Test$$
#
# $index jacobian, example$$
# $index derivative, entire example$$
# $index entire, derivative example$$
# $index example, entire derivative$$
#
# $code
# $verbatim%example/jacobian.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
# Example using a_float -----------------------------------------------------
def pycppad_test_jacobian():
  delta = 10. * numpy.finfo(float).eps
  x     = numpy.array( [ 0., 0. ] )
  a_x   = independent(x)
  a_y   = numpy.array( [ 
    a_x[0] * exp(a_x[1]) , 
    a_x[0] * sin(a_x[1]) ,
    a_x[0] * cos(a_x[1]) 
  ] )
  f   = adfun(a_x, a_y)
  x   = numpy.array( [ 2., 3. ] )
  J   = f.jacobian(x)
  assert abs( J[0,0] -        exp(x[1]) ) < delta
  assert abs( J[0,1] - x[0] * exp(x[1]) ) < delta
  assert abs( J[1,0] -        sin(x[1]) ) < delta
  assert abs( J[1,1] - x[0] * cos(x[1]) ) < delta
  assert abs( J[2,0] -        cos(x[1]) ) < delta
  assert abs( J[2,1] + x[0] * sin(x[1]) ) < delta
# Example using a2float -----------------------------------------------------
def pycppad_test_jacobian_a2():
  delta = 10. * numpy.finfo(float).eps
  a_x   = ad( numpy.array( [ 0., 0. ] ) )
  a2x   = independent(a_x)
  a2y   = numpy.array( [ 
    a2x[0] * exp(a2x[1]) , 
    a2x[0] * sin(a2x[1]) ,
    a2x[0] * cos(a2x[1]) 
  ] )
  a_f   = adfun(a2x, a2y)
  x     = numpy.array( [2., 3.] )
  a_x   = ad(x)
  a_J   = a_f.jacobian(a_x)
  assert abs( a_J[0,0] -        exp(x[1]) ) < delta
  assert abs( a_J[0,1] - x[0] * exp(x[1]) ) < delta
  assert abs( a_J[1,0] -        sin(x[1]) ) < delta
  assert abs( a_J[1,1] - x[0] * cos(x[1]) ) < delta
  assert abs( a_J[2,0] -        cos(x[1]) ) < delta
  assert abs( a_J[2,1] + x[0] * sin(x[1]) ) < delta
# END CODE
