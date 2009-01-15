# First order reverse mode with two levels of taping

from cppad import *
def test_reverse_a2():
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

  # derivative of f with respect to u (using a_float)
  p  = 1                         # order of derivative
  w  = numpy.array( [ ad(1) ] )        # vector of weights
  fp = f.reverse(p, w)           # derivative of f w.r.t u
  assert fp[0] == 2. * u[1]      # f_u0(u0, u1) = 2. * u1
  assert fp[1] == 2. * u[0]      # f_u1(u0, u1) = 2. * u0

  # stop a_float recording and store operations if g
  a_y = 2. * fp                  # g(x0, x1) = 2 * f_u (x0, 2 * x1)
  g   = adfun(a_x, a_y)          #           = [ 8 * x1 , 4 * x0  ]  

  # evaluate the function g(x) at x = (4, 5) using float operations
  p  = 0
  x  = numpy.array( [ 4. , 5. ] )
  gp = g.forward(p, x)
  assert gp[0] == 8. * x[1]
  assert gp[1] == 4. * x[0]

  # derivative of the first component of g with respect to x (using float)
  p  = 1
  w = numpy.array( [ 1. , 0. ] )
  gp = g.reverse(p, w)
  assert gp[0] == 0.
  assert gp[1] == 8.
