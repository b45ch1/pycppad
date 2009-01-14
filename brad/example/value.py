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
  x   = 2
  a_x = ad(x)
  a2x = ad(a_x)
  #
  assert type(value(a_x)) == float   and x == value(a_x)
  assert type(value(a2x)) == a_float and x == value(a2x)
  #
  x   = numpy.array( [ 1 , 2 , 3 ] )
  a_x = ad(x)
  a2x = ad(a_x)
  #
  for i in range( len(a_x) ) :  
    xi = value(a_x[i])
    assert type(xi) == float and xi == x[i]
  for i in range( len(a2x) ) :  
    a_xi = value(a2x[i])
    assert type(a_xi) == a_float and a_xi == x[i]
# END CODE
