
# First order forward mode with two levels of taping
from pyad.cppad import *
def test_forward_a2():
  # start recording a_float operations
  x   = numpy.array( [ 2. , 3. ] )
  a_x = independent(x)

  # start recording a2float operations
  u   = numpy.array( [ a_x[0] , ad(4) ] )
  a_u = independent(u)

  # stop a2float recording and store operations if f
  a_v = numpy.array( [ 2. * a_u[0] * a_u[1] ] )
  f   = adfun(a_u, a_v)              # f(u0, u1) = 2. * u0 * u1

  # evaluate the function f(u) using a_float operations
  u   = numpy.array([a_x[0] , 2.*a_x[1]])  # u0 = x0, u1 = 2 * x1
  p   = 0
  fp  = f.forward(p, u)
  assert fp[0] == 2. * u[0] * u[1]

  # evaluate partial of f with respect to the second component (using a_float)
  p  = 1
  up = numpy.array( [ ad(0) , ad(1) ] )
  fp = f.forward(p, up)
  assert fp[0] == 2. * u[0]         # f_u1(u0, u1) = 2. * u0

  # stop a_float recording and store operations if g
  a_y = 2. * fp
  g   = adfun(a_x, a_y)          # g(x0, x1) = 2. * f_u1(x0, 2 * x1) = 4 * x0

  # evaluate the function g(x) at x = (4, 5) using float operations
  p  = 0
  x  = numpy.array( [ 4. , 5. ] )
  gp = g.forward(p, x)
  assert gp[0] == 4. * x[0]

  # evaluate the partial of g with respect to x0 (using float)
  p  = 1
  xp = numpy.array( [ 1. , 0. ] )
  gp = g.forward(p, xp)
  assert gp[0] == 4.

  # evaluate the partial of g with respect to x1 (using float)
  p  = 1
  xp = numpy.array( [ 0. , 1. ] )
  gp = g.forward(p, xp)
  assert gp[0] == 0.
