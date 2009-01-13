# $begin ad_binary_a2.py$$ $newlinech #$$
#
# $section Binary Operators With an a2float Result: Example and Test$$
#
# $index +, example$$
# $index -, example$$
# $index *, example$$
# $index /, example$$
# $index **, example$$
#
# $index example, +$$
# $index example, -$$
# $index example, *$$
# $index example, /$$
# $index example, **$$
#
# $index binary, a2float operator example$$
# $index operator, a2float binary example$$
# $index example, a2float binary operator$$
# $index a2float, binary operator$$
#
# $code
# $verbatim%example/ad_binary_a2.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
def test_ad_binary_a2() :
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2., -2., 0.,  4.,   4. ]
  y_list = [ -2,   2,  2., .5,   -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a2x = ad( ad(x) )
    a2y = ad( ad(y) )
    #
    assert abs( a2x + a2y - (x + y) ) < delta
    assert abs( a2x + y   - (x + y) ) < delta
    assert abs( x   + a2y - (x + y) ) < delta
    #
    assert abs( a2x - a2y - (x - y) ) < delta
    assert abs( a2x - y   - (x - y) ) < delta
    assert abs( x   - a2y - (x - y) ) < delta
    #
    assert abs( a2x * a2y - x * y ) < delta
    assert abs( a2x * y   - x * y ) < delta
    assert abs( x   * a2y - x * y ) < delta
    #
    assert abs( a2x / a2y - x / y ) < delta
    assert abs( a2x / y   - x / y ) < delta
    assert abs( x   / a2y - x / y ) < delta
    #
    assert abs( a2x ** a2y - x ** y ) < delta
    assert abs( a2x ** y   - x ** y ) < delta
    assert abs( x   ** a2y - x ** y ) < delta
  #
  a_x = array( [ ad(-2) , ad(+2) ] )
  a2x = independent(a_x)
  n   = 3.
  a2y = a2x ** n
  a_f = adfun(a2x, a2y)
  J   = a_f.jacobian(a_x)
  for j in range( len(a2x) ) :
    for i in range( len(a2y) ) :
      if i == j : assert abs( J[i][j] - n * a_x[j] ** (n-1) ) < delta
      else :      J[i][j] == 0.
# END CODE
