# the C++ standard math functions

# Not completed
# Working on this test and implementation of standard math functions

import math
def test_std_math(): 
   n     = 10
   delta = 10. * numpy.finfo(float).eps
   pi    = numpy.pi
   x     = pi / 6
   a_x   = ad(x)
   assert a_x.sin() == math.sin(x)
