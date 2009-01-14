# $begin compare_op_a2.py$$ $newlinech #$$
# $spell
#	bool
# $$
#
#
# $section a2float Comparison Operators: Example and Test$$
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
# $index bool, a2float binary operator example$$
# $index binary, bool a2float operator example$$
# $index operator, bool a2float binary example$$
# $index example, bool a2float binary operator$$
# $index comparison, a2float operator$$
# $index a2float, comparison operator$$
#
# $code
# $verbatim%example/compare_op_a2.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
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
