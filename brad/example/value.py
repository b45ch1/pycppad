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
  a_x = ad(2)
  a2x = ad(a_x)
  #
  assert type(value(a_x)) == float   and 2. == value(a_x)
  assert type(value(a2x)) == a_float and 2. == value(a2x)
# END CODE
