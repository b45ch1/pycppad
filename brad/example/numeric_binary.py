# $begin numeric_binary.py$$ $newlinech #$$
#
# $section a_float Numeric Binary Operators: Example and Test$$
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
# $index numeric, binary operator example$$
# $index binary, numeric operator example$$
# $index operator, numeric binary example$$
# $index example, numeric binary operator$$
#
# $code
# $verbatim%example/numeric_binary.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
def test_numeric_binary() :
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2., -2., 0.,  4.,   4. ]
  y_list = [ -2,   2,  2., .5,   -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a_x = ad(x)
    a_y = ad(y)
    #
    assert abs( a_x + a_y - (x + y) ) < delta
    assert abs( a_x + y   - (x + y) ) < delta
    assert abs( x   + a_y - (x + y) ) < delta
    #
    assert abs( a_x - a_y - (x - y) ) < delta
    assert abs( a_x - y   - (x - y) ) < delta
    assert abs( x   - a_y - (x - y) ) < delta
    #
    assert abs( a_x * a_y - x * y ) < delta
    assert abs( a_x * y   - x * y ) < delta
    assert abs( x   * a_y - x * y ) < delta
    #
    assert abs( a_x / a_y - x / y ) < delta
    assert abs( a_x / y   - x / y ) < delta
    assert abs( x   / a_y - x / y ) < delta
    #
    assert abs( a_x ** a_y - x ** y ) < delta
    assert abs( a_x ** y   - x ** y ) < delta
    assert abs( x   ** a_y - x ** y ) < delta
  #
  x   = array( [ -2 , +2 ] )
  a_x = independent(x)
  n   = 3
  a_y = a_x ** n
  f   = adfun(a_x, a_y)
  J   = f.jacobian(x)
  print J
  for j in range( len(a_x) ) :
    for i in range( len(a_y) ) :
      if i == j : assert abs( J[i][j] - n * x[j] ** (n-1) ) < delta
      else :      assert J[i][j] == 0.
# END CODE
