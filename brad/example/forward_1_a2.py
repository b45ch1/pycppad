# First order forward mode with two levels of taping

def test_forward_1_a2():
  # start recording a_double operations
  a_x = array( [ ad(2) , ad(3) ] )
  independent(a_x)

  # start recording a2double operations
  a_u = array( [ ad(a_x[0]) , ad(ad(4)) ] )
  independent(a_u)

  # stop a2double recording and store operations if f
  a_v = array( [ 2. * a_u[0] * a_u[1] ] )
  f   = adfun(a_u, a_v)              # f(u0, u1) = 2. * u0 * u1

  # evaluate the function f(u) using a_double operations
  u   = array([a_x[0] , 2.*a_x[1]])  # u0 = x0, u1 = 2 * x1
  p   = 0
  fp  = f.forward(p, u)
  assert fp == 2. * u[0] * u[1]

  # evaluate partial of f with respect to the second component (using a_double)
  p  = 1
  up = array( [ ad(0) , ad(1) ] )
  fp = f.forward(p, up)
  assert fp == 2. * u[0]         # f_u1(u0, u1) = 2. * u0

  # stop a_double recording and store operations if g
  a_y = 2. * fp
  g   = adfun(a_x, a_y)          # g(x0, x1) = 2. * f_u1(x0, 2 * x1) = 4 * x0

  # evaluate the function g(x) at x = (4, 5) using double operations
  p  = 0
  x  = array( [ 4. , 5. ] )
  gp = g.forward(p, x)
  assert gp == 4. * x[0]

  # evaluate the partial of g with respect to x0 (using double)
  p  = 1
  xp = array( [ 1. , 0. ] )
  gp = g.forward(p, xp)
  assert gp == 4.

  # evaluate the partial of g with respect to x1 (using double)
  p  = 1
  xp = array( [ 0. , 1. ] )
  gp = g.forward(p, xp)
  assert gp == 0.
