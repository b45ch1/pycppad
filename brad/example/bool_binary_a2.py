# $begin bool_binary_a2.py$$ $newlinech #$$
# $spell
#	bool
# $$
#
#
# $section a2float Binary Operators With a Boolean Result: Example and Test$$
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
# $index bool, a2float binary operator example$$
# $index binary, bool a2float operator example$$
# $index operator, bool a2float binary example$$
# $index example, bool a2float binary operator$$
#
# $code
# $verbatim%example/bool_binary.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_bool_binary_a2() :
  delta = 10. * numpy.finfo(float).eps
  x_array = numpy.array( range(5) )
  y_array = 6. - x_array
  for i in range( len(x_array) ) :
    x   = x_array[i]
    y   = y_array[i]
    a2x = ad(ad(x))
    a2y = ad(ad(y))
    #
    assert (a2x < a2y)  == ( x < y)
    assert (a2x > a2y)  == ( x > y)
    assert (a2x <= a2y) == ( x <= y)
    assert (a2x >= a2y) == ( x >= y)
    assert (a2x == a2y) == ( x == y)
    assert (a2x != a2y) == ( x != y)
  #
  n        = 3.
  x        = numpy.array( [ -2 , +2 ] )
  a_x      = numpy.array( [ ad(x[0]) , ad(x[1]) ] )
  a2x      = independent(x)
  positive = a2x >= 0 
  # At some level, each element of positive is being converted to a float 
  # before interfacing to pycppad * operator.
  a2y      = ( a2x ** n ) * positive
  f        = adfun(a2x, a2y)
  J        = f.jacobian(x)
  for j in range( len(a2x) ) :
    for i in range( len(a2y) ) :
      if i == j  and x[i] >= 0 :
        assert abs( J[i][j] - n * x[j] ** (n-1) ) < delta
      else :
        assert J[i][j] == 0.
# END CODE
