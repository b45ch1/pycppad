# $begin div_op.py$$ $newlinech #$$
#
# $section Division Operator: Example and Test$$
#
# $index /, example$$
#
# $index example, /$$
#
# $index division, division operator example$$
# $index operator, division example$$
# $index example, division operator$$
#
# $code
# $verbatim%example/div_op.py%0%# BEGIN CODE%# END CODE%1%$$
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
  assert tmp3 == x / y
# END CODE

pycppad_test_div_op()
