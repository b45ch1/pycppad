# $begin bool_binary.py$$ $newlinech #$$
# $spell
#	bool
# $$
#
#
# $section a_float Binary Operators With a Boolean Result: Example and Test$$
#
# $index > example$$
# $index < example$$
# $index >= example$$
# $index <= example$$
# $index == example$$
# $index != example$$
#
# $index example, >$$
# $index example, <$$
# $index example, >=$$
# $index example, <=$$
# $index example, ==$$
# $index example, !=$$
#
# $index bool, a_float binary operator example$$
# $index binary, bool a_float operator example$$
# $index operator, bool a_float binary example$$
# $index example, bool a_float binary operator$$
#
# $code
# $verbatim%example/bool_binary.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_bool_binary() :
  delta = 10. * numpy.finfo(float).eps
  x_array = numpy.array( range(5) )
  y_array = 6. - x_array
  for i in range( len(x_array) ) :
    x   = x_array[i]
    y   = y_array[i]
    a_x = ad(x)
    a_y = ad(y)
    #
    assert (a_x < a_y)  == ( x < y )
    assert (a_x > a_y)  == ( x > y )
    assert (a_x <= a_y) == ( x <= y )
    assert (a_x >= a_y) == ( x >= y )
    assert (a_x == a_y) == ( x == y )
    assert (a_x != a_y) == ( x != y )
  #
  n        = 3.
  x        = numpy.array( [ -2 , +2 ] )
  a_x      = independent(x)
  positive = a_x >= 0 
  # At some level, each element of positive is being converted to a float 
  # before interfacing to pycppad * operator.
  a_y      = ( a_x ** n ) * positive
  f        = adfun(a_x, a_y)
  J        = f.jacobian(x)
  for j in range( len(a_x) ) :
    for i in range( len(a_y) ) :
      if i == j  and x[i] >= 0 :
        assert abs( J[i][j] - n * x[j] ** (n-1) ) < delta
      else :
        assert J[i][j] == 0.
# END CODE
