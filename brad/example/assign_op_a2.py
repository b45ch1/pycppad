# $begin assign_op_a2.py$$ $newlinech #$$
#
# $section Computed Assignment Operators Using a2float: Example and Test$$
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
# $index a2float, computed assignment operator example$$
# $index computed, a2float computed assignment operator example$$
# $index assignment, a2float assignment operator example$$
# $index operator, a2float computed operator example$$
# $index example, a2float computed assignment operator$$
#
# $code
# $verbatim%example/assign_op_a2.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_assign_op_a2() :
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2., -2., 0.,  4.,   4. ]
  y_list = [ -2,   2,  2., .5,   -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a2y = ad(ad(y))
    #
    tmp = ad(ad(x))
    tmp += a2y
    assert abs( tmp - (x + y) ) < delta
    #
    tmp = ad(ad(x))
    tmp -= y
    assert abs( tmp  - (x - y) ) < delta
    #
    tmp = x
    tmp *= a2y
    assert abs( tmp - x * y ) < delta
    #
    tmp = ad(ad(x))
    tmp /= a2y
    assert abs( tmp - x / y ) < delta
    #
  #
  a_x   = numpy.array( [ -1 * ad(2) , ad(2) ] )
  a2x = independent(a_x)
  a2y = a2x + 2. 
  a2y *= 5.
  f   = adfun(a2x, a2y)
  J   = f.jacobian(a_x)
  for j in range( len(a2x) ) :
    for i in range( len(a2y) ) :
      if i == j : assert abs( J[i][j] - 5. ) < delta
      else :      assert J[i][j] == 0.
# END CODE
