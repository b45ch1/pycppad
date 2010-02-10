# $begin ad_unary.py$$ $newlinech #$$
#
# $section Unary Plus and Minus Operators: Example and Test$$
#
# $index +, unary$$
# $index -, unary$$
#
# $index example, +$$
# $index example, -$$
# $index unary, +$$
# $index unary, -$$
#
# $code
# $verbatim%example/ad_unary.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE

from pycppad import *
import numpy
# Example using a_float ------------------------------------------------------
def pycppad_test_ad_unary() :
  x       = ad(2.)
  plus_x  = + x
  minus_x = - x
  # test using corresponding unary float operators 
  assert value(plus_x)  == + value(x)
  assert value(minus_x) == - value(x)
  #
  x       = ad( numpy.array( [ 1. , 2. ] ) )
  plus_x  = + x
  minus_x = - x
  # test using corresponding unary float operators 
  assert numpy.all( value(plus_x)  == + value(x) )
  assert numpy.all( value(minus_x) == - value(x) )

# Example using a2float ------------------------------------------------------
def pycppad_test_ad_unary_a2() :
  x       = ad( ad(2.) )
  plus_x  = + x
  minus_x = - x
  # test using corresponding unary a_float operators 
  assert value(plus_x)  == + value(x)
  assert value(minus_x) == - value(x)
  #
  x       = ad( ad( numpy.array( [ 1. , 2. ] ) ) )
  plus_x  = + x
  minus_x = - x
  # test using corresponding unary float operators 
  assert numpy.all( value(plus_x)  == + value(x) )
  assert numpy.all( value(minus_x) == - value(x) )

# END CODE

