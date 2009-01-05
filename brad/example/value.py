# $begin value.py$$ $newlinech #$$
#
# $section value: Example and Test$$
#
# $index value, example$$
# $index example, value$$
#
# $code
# $verbatim%example/value.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_value() :
  x   = float(2)
  a_x = ad(x)
  a2x = ad(a_x)
  #
  assert type(x)   == type(value(a_x)) and x   == value(a_x)
  assert type(a_x) == type(value(a2x)) and a_x == value(a2x)
# END CODE
