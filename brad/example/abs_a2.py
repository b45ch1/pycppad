# Note that derivatives of the abs function at zero depend on the direction
#
def test_abs_a2() :
  x   = array( [ ad(-1),  ad(0),  ad(1)] )
  n   = len(x)
  a_x = independent(x)
  a_y = abs( a_x )
  f   = adfun(a_x, a_y)
  f.forward(0, x)
  dx  = array( list( ad(0) for i in range(n) ) )
  for i in range( n ) :
    dx[i] = ad(0)
  for i in range( n ) :
    dx[i] = ad(1)
    df    = f.forward(1, dx)
    if x[i] >= 0 :
      assert df[i] == +1.
    else :
      assert df[i] == -1.
    dx[i] = ad(-1)
    df    = f.forward(1, dx)
    if x[i] > 0 :
      assert df[i] == -1.
    else :
      assert df[i] == +1.
    dx[i] = ad(0)
