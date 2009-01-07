# $begin pow_a2.py$$ $newlinech #$$
#
# $section exponentiation: Example and Test$$
#
# $index pow, example$$
# $index example, pow$$
# $index exponentiation, example$$
# $index example, exponentiation$$
#
# $code
# $verbatim%example/pow_a2.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
def test_pow_a2() :
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2, -2., 0,  4,   4 ]
  y_list = [ -2,   2, 2, .5, -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a_x = ad(ad(x))
    a_y = ad(ad(y))
    assert abs( a_x ** a_y - x ** y ) < delta
    assert abs( a_x ** y   - x ** y ) < delta
    assert abs( x   ** a_y - x ** y ) < delta
  #
  x   = array( [ ad(-2) ] )
  a_x = independent(x)
  n   = 3
  a_y = a_x ** n
  f   = adfun(a_x, a_y)
  J   = f.jacobian(x)
  assert abs( J[0][0] - n * x[0] ** (n-1) ) < delta
# END CODE
