# $begin hessian.py$$ $newlinech #$$
# $spell
#	Hessian
# $$
#
# $section Hessian Driver: Example and Test$$
#
# $index hessian, example$$
# $index example, hessian$$
#
# $code
# $verbatim%example/hessian.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
# Example using a_float -----------------------------------------------------
def pycppad_test_hessian():
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
  w   = numpy.array( [ 0., 1., 0. ] ) # compute Hessian of x0 * sin(x1)
  H   = f.hessian(x, w)
  assert abs( H[0,0] - 0.               ) < delta
  assert abs( H[0,1] - cos(x[1])        ) < delta
  assert abs( H[1,0] - cos(x[1])        ) < delta
  assert abs( H[1,1] + x[0] * sin(x[1]) ) < delta
# Example using a2float -----------------------------------------------------
def pycppad_test_hessian_a2():
  delta = 10. * numpy.finfo(float).eps
  a_x   = ad( numpy.array( [ 0., 0. ] ) )
  a2x   = independent(a_x)
  a2y   = numpy.array( [ 
    a2x[0] * exp(a2x[1]) , 
    a2x[0] * sin(a2x[1]) ,
    a2x[0] * cos(a2x[1]) 
  ] )
  a_f = adfun(a2x, a2y)
  x   = numpy.array( [ 2., 3. ] )
  a_x = ad(x)
  a_w = ad( numpy.array( [ 0., 1., 0. ] ) ) # compute Hessian of x0 * sin(x1)
  a_H = a_f.hessian(a_x, a_w)
  assert abs( a_H[0,0] - 0.               ) < delta
  assert abs( a_H[0,1] - cos(x[1])        ) < delta
  assert abs( a_H[1,0] - cos(x[1])        ) < delta
  assert abs( a_H[1,1] + x[0] * sin(x[1]) ) < delta
# END CODE
