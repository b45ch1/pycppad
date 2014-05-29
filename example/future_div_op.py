# $begin future_div_op.py$$ $newlinech #$$
#
# $section Future Division Operator: Example and Test$$
# $index future /, example$$
# $index example, future /$$
# $index /, future example$$
#
# $code
# $verbatim%example/future_div_op.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
# Example using a_float ------------------------------------------------------
from __future__ import division
from pycppad import *
def pycppad_test_div_op() :
  x = 2.
  y = 3.
  #
  tmp1 = ad(x)
  tmp2 = ad(y)
  tmp3 = tmp1/tmp2
  tmp4 = x/tmp2
  tmp5 = tmp1/y

  assert tmp3 == x / y
  assert tmp4 == x / y
  assert tmp5 == x / y

# END CODE

pycppad_test_div_op()
