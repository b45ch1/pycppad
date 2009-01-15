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
# Example using a_float ------------------------------------------------------
def test_value() :
  x   = 2
  a_x = ad(x)
  #
  assert type(value(a_x)) == float   and value(a_x) == x
  #
  x   = numpy.array( [ 1 , 2 , 3 ] )
  a_x = ad(x)
  #
  for i in range( len(a_x) ) :  
    xi = value(a_x[i])
    assert type(xi) == float and xi == x[i]

# Example using a2float ------------------------------------------------------
def test_value_a2() :
  x   = 2
  a2x = ad(ad(x))
  #
  assert type(value(a2x)) == a_float and value(a2x) == x
  #
  x   = numpy.array( [ 1 , 2 , 3 ] )
  a2x = ad(ad(x))
  #
  for i in range( len(a2x) ) :  
    a_xi = value(a2x[i])
    assert type(a_xi) == a_float and a_xi == x[i]
# END CODE
