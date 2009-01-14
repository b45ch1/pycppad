# $begin assign_op.py$$ $newlinech #$$
#
# $section Computed Assignment Operators Using a_float: Example and Test$$
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
# $index a_float, computed assignment operator example$$
# $index computed, a_float computed assignment operator example$$
# $index assignment, a_float assignment operator example$$
# $index operator, a_float computed operator example$$
# $index example, a_float computed assignment operator$$
#
# $code
# $verbatim%example/assign_op.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_assign_op() :
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2., -2., 0.,  4.,   4. ]
  y_list = [ -2,   2,  2., .5,   -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a_y = ad(y)
    #
    tmp = ad(x)
    tmp += a_y
    assert abs( tmp - (x + y) ) < delta
    #
    tmp = ad(x)
    tmp -= y
    assert abs( tmp  - (x - y) ) < delta
    #
    tmp = x
    tmp *= a_y
    assert abs( tmp - x * y ) < delta
    #
    tmp = ad(x)
    tmp /= a_y
    assert abs( tmp - x / y ) < delta
    #
  #
  x   = array( [ -2 , +2 ] )
  a_x = independent(x)
  a_y = a_x + 2. 
  a_y *= 5.
  f   = adfun(a_x, a_y)
  J   = f.jacobian(x)
  for j in range( len(a_x) ) :
    for i in range( len(a_y) ) :
      if i == j : assert abs( J[i][j] - 5. ) < delta
      else :      assert J[i][j] == 0.
# END CODE
