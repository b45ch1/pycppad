# Note that derivatives of the abs function at zero depend on the direction
#
def test_abs() :
  x   = array( [ -1.,  0.,  1.] )
  n   = len(x)
  a_x = independent(x)
  a_y = abs( a_x )
  f   = adfun(a_x, a_y)
  f.forward(0, x)
  dx  = numpy.zeros(n, dtype=float)
  for i in range( n ) :
    dx[i] = 1.
    df    = f.forward(1, dx)
    if x[i] >= 0 :
      assert df[i] == +1.
    else :
      assert df[i] == -1.
    dx[i] = -1.
    df    = f.forward(1, dx)
    if x[i] > 0 :
      assert df[i] == -1.
    else :
      assert df[i] == +1.
    dx[i] = 0.
