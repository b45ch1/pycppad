# $begin assign_op.py$$ $newlinech #$$
#
# $section Computed Assignment Operators: Example and Test$$
#
# $index +=, example$$
# $index -=, example$$
# $index *=, example$$
# $index /=, example$$
#
# $index example, +=$$
# $index example, -=$$
# $index example, *=$$
# $index example, /=$$
#
# $index computed, assignment operator example$$
# $index assignment, assignment operator example$$
# $index operator, computed assignment example$$
# $index example, computed assignment operator$$
#
# $code
# $verbatim%example/assign_op.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
# Example using a_float ------------------------------------------------------
from pyad import *
def test_assign_op() :
  x = 2.
  y = 3.
  #
  tmp  = ad(x)
  tmp += ad(y)
  assert tmp == x + y
  tmp  = ad(x)
  tmp += y
  assert tmp == x + y
  #
  tmp  = ad(x)
  tmp -= ad(y)
  assert tmp == x - y
  tmp  = ad(x)
  tmp -= y
  assert tmp == x - y
  #
  tmp  = ad(x)
  tmp *= ad(y)
  assert tmp == x * y
  tmp  = ad(x)
  tmp *= y
  assert tmp == x * y
  #
  tmp  = ad(x)
  tmp /= ad(y)
  assert tmp == x / y
  tmp  = ad(x)
  tmp /= y
  assert tmp == x / y

# Example using a2float ------------------------------------------------------
from pyad import *
def test_assign_op_a2() :
  x = 2.
  y = 3.
  #
  tmp  = ad(ad(x))
  tmp += ad(ad(y))
  assert tmp == x + y
  tmp  = ad(ad(x))
  tmp += y
  assert tmp == x + y
  #
  tmp  = ad(ad(x))
  tmp -= ad(ad(y))
  assert tmp == x - y
  tmp  = ad(ad(x))
  tmp -= y
  assert tmp == x - y
  #
  tmp  = ad(ad(x))
  tmp *= ad(ad(y))
  assert tmp == x * y
  tmp  = ad(ad(x))
  tmp *= y
  assert tmp == x * y
  #
  tmp  = ad(ad(x))
  tmp /= ad(ad(y))
  assert tmp == x / y
  tmp  = ad(ad(x))
  tmp /= y
  assert tmp == x / y
# END CODE
