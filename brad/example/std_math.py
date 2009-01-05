# the C++ standard math functions
#
# Note that numpy uses the non-standard names arccos, arcsin, and arctan

import math
def test_std_math(): 
   n     = 10
   delta = 10. * numpy.finfo(float).eps
   pi    = numpy.pi
   x     = pi / 6
   a_x   = ad(x)
   a2x   = ad(a_x)

   # all the a_float unary standard math functions
   assert abs( arccos(a_x) - math.acos(x) )  < delta
   assert abs( arcsin(a_x) - math.asin(x) )  < delta
   assert abs( arctan(a_x) - math.atan(x) )  < delta
   assert abs( cos(a_x)    - math.cos(x) )   < delta
   assert abs( cosh(a_x)   - math.cosh(x) )  < delta
   assert abs( exp(a_x)    - math.exp(x) )   < delta
   assert abs( log(a_x)    - math.log(x) )   < delta
   assert abs( log10(a_x)  - math.log10(x) ) < delta
   assert abs( sin(a_x)    - math.sin(x) )   < delta
   assert abs( sinh(a_x)   - math.sinh(x) )  < delta
   assert abs( sqrt(a_x)   - math.sqrt(x) )  < delta
   assert abs( tan(a_x)    - math.tan(x) )   < delta
   assert abs( tanh(a_x)   - math.tanh(x) )  < delta

   # all the a2float unary standard math functions
   assert abs( arccos(a2x) - math.acos(x) )  < delta
   assert abs( arcsin(a2x) - math.asin(x) )  < delta
   assert abs( arctan(a2x) - math.atan(x) )  < delta
   assert abs( cos(a2x)    - math.cos(x) )   < delta
   assert abs( cosh(a2x)   - math.cosh(x) )  < delta
   assert abs( exp(a2x)    - math.exp(x) )   < delta
   assert abs( log(a2x)    - math.log(x) )   < delta
   assert abs( log10(a2x)  - math.log10(x) ) < delta
   assert abs( sin(a2x)    - math.sin(x) )   < delta
   assert abs( sinh(a2x)   - math.sinh(x) )  < delta
   assert abs( sqrt(a2x)   - math.sqrt(x) )  < delta
   assert abs( tan(a2x)    - math.tan(x) )   < delta
   assert abs( tanh(a_x)   - math.tanh(x) )  < delta

   # example derivative calculation
   x = array( [ x ] )
   a_x = independent(x)
   a_y = sin(a_x)
   f   = adfun(a_x, a_y)
   J   = f.jacobian(x)
   assert abs( J[0] - cos(x[0]) ) < delta
