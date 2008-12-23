#! /usr/bin/python
#

import numpy
from cppad import *

def test_array_element_type_is_int():
  a_x = numpy.array( [ ad(1.) ] )
  independent(a_x)
  a_y = a_x
  f = adfun(a_x, a_y) 
  x = numpy.array( [ 1 ] )
  y = f.forward(0, x) 
  assert y[0] == 1


def test_normal_standard_math_syntax_not_supported():
  a_x = ad(1.)
  a_y = a_x.sin()    # supported
  a_z = sin(a_x)     # generates an error

