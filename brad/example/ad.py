# $begin ad.py$$ $newlinech #$$
#
# $section ad: Example and Test$$
#
# $index ad, example$$
# $index example, ad$$
#
# $code
# $verbatim%example/ad.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_ad() :
  x   = 1
  a_x = ad(x)
  a2x = ad(a_x)
  #
  assert type(a_x) == a_float and a_x == x
  assert type(a2x) == a2float and a2x == x
# END CODE
