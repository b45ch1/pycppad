# $begin compare_op.py$$ $newlinech #$$
# $spell
#	bool
# $$
#
#
# $section a_float Comparison Operators: Example and Test$$
#
# $index >, example$$
# $index <, example$$
# $index >=, example$$
# $index <=, example$$
# $index ==, example$$
# $index !=, example$$
#
# $index example, >$$
# $index example, <$$
# $index example, >=$$
# $index example, <=$$
# $index example, ==$$
# $index example, !=$$
#
# $index bool, a_float comparison operator example$$
# $index binary, bool a_float operator example$$
# $index operator, bool a_float binary example$$
# $index example, bool a_float binary operator$$
# $index comparison, a_float operator$$
# $index a_float, comparison operator$$
#
# $code
# $verbatim%example/compare_op.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
# Example using a_float ------------------------------------------------------
def test_compare_op():
  x = ad(2.)
  y = ad(3.)
  z = ad(2.)
  
  # assert comparisons that should be true
  assert x == x
  assert x == z
  assert x != y
  assert x <= x
  assert x <= z
  assert x <= y
  assert x <  y
  
  # assert comparisons that should be false
  assert not x == y
  assert not x != z
  assert not x != x
  assert not x >= y
  assert not x >  y
# Example using a2float ------------------------------------------------------
def test_compare_op_a2():
  x = ad(ad(2.))
  y = ad(ad(3.))
  z = ad(ad(2.))
  
  # assert comparisons that should be true
  assert x == x
  assert x == z
  assert x != y
  assert x <= x
  assert x <= z
  assert x <= y
  assert x <  y
  
  # assert comparisons that should be false
  assert not x == y
  assert not x != z
  assert not x != x
  assert not x >= y
  assert not x >  y
# END CODE
