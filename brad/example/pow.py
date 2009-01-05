# The ** operator in python corresponds to the pow function in C++

def test_pow() :
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2, -2., 0,  4,   4 ]
  y_list = [ -2,   2, 2, .5, -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a_x = ad(x)
    a_y = ad(y)
    assert abs( a_x ** a_y - x ** y ) < delta
    assert abs( a_x ** y   - x ** y ) < delta
    # This case does not yet work
    # assert abs( x   ** a_y - x ** y ) < delta
  #
  x   = array( [ -2 ] )
  a_x = independent(x)
  n   = 3
  a_y = a_x ** n
  f   = adfun(a_x, a_y)
  J   = f.jacobian(x)
  assert abs( J[0] - n * x[0] ** (n-1) ) < delta
