from cppad import *
# Comparision operators with type a2double

def test_compare_op_a2():
	x = ad(ad(2.))
	y = ad(ad(3.))
	z = ad(ad(2.))
	
	# assert comparisions that should be true
	assert x == x
	assert x == z
	assert x != y
	assert x <= x
	assert x <= z
	assert x <= y
	assert x <  y
	
	# assert omparisions that should be false
	assert not x == y
	assert not x != z
	assert not x != x
	assert not x >= y
	assert not x >  y
# Comparison operators with type a_double

def test_compare_op():
	x = ad(2.)
	y = ad(3.)
	z = ad(2.)
	
	# assert comparisions that should be true
	assert x == x
	assert x == z
	assert x != y
	assert x <= x
	assert x <= z
	assert x <= y
	assert x <  y
	
	# assert omparisions that should be false
	assert not x == y
	assert not x != z
	assert not x != x
	assert not x >= y
	assert not x >  y
# First order forward mode with two levels of taping

def test_forward_1_a2():
  # start recording a_double operations
  x   = array( [ 2. , 3. ] )
  a_x = independent(x)

  # start recording a2double operations
  u   = array( [ a_x[0] , ad(4) ] )
  a_u = independent(u)

  # stop a2double recording and store operations if f
  a_v = array( [ 2. * a_u[0] * a_u[1] ] )
  f   = adfun(a_u, a_v)              # f(u0, u1) = 2. * u0 * u1

  # evaluate the function f(u) using a_double operations
  u   = array([a_x[0] , 2.*a_x[1]])  # u0 = x0, u1 = 2 * x1
  p   = 0
  fp  = f.forward(p, u)
  assert fp[0] == 2. * u[0] * u[1]

  # evaluate partial of f with respect to the second component (using a_double)
  p  = 1
  up = array( [ ad(0) , ad(1) ] )
  fp = f.forward(p, up)
  assert fp[0] == 2. * u[0]         # f_u1(u0, u1) = 2. * u0

  # stop a_double recording and store operations if g
  a_y = 2. * fp
  g   = adfun(a_x, a_y)          # g(x0, x1) = 2. * f_u1(x0, 2 * x1) = 4 * x0

  # evaluate the function g(x) at x = (4, 5) using double operations
  p  = 0
  x  = array( [ 4. , 5. ] )
  gp = g.forward(p, x)
  assert gp[0] == 4. * x[0]

  # evaluate the partial of g with respect to x0 (using double)
  p  = 1
  xp = array( [ 1. , 0. ] )
  gp = g.forward(p, xp)
  assert gp[0] == 4.

  # evaluate the partial of g with respect to x1 (using double)
  p  = 1
  xp = array( [ 0. , 1. ] )
  gp = g.forward(p, xp)
  assert gp[0] == 0.
# First order forward mode.

def test_forward_1():

  # start record a_double operations
  x   = array( [ 2., 3. ] )  # value of independent variables
  a_x = independent(x)       # declare independent variables

  # stop recording and store operations in the function object f
  a_y = array( [ 2. * a_x[0] * a_x[1] ] ) # dependent variables
  f   = adfun(a_x, a_y)                   # f(x0, x1) = 2 * x0 * x1

  # evaluate the function at a different argument value
  p  = 0                                 # order zero for function values
  x  = array( [ 3. , 4. ] )              # argument value
  fp = f.forward(p, x)                   # function value
  assert fp[0] == 2. * x[0] * x[1]       # f(x0, x1) = 2 * x0 * x1

  # evalute partial derivative of f(x0, x1) with respect to x0
  p  = 1                                 # order one for first derivatives
  xp = array( [ 1. , 0. ] )              # direction for differentiation
  fp = f.forward(p, xp)                  # value of directional derivative
  assert fp[0] == 2. * x[1]              # f_x0 (x0, x1) = 2 * x1

  # evalute partial derivative of f(x0, x1) with respect to x
  p  = 1
  xp = array( [ 0. , 1. ] )              # the x1 direction
  fp = f.forward(p, xp)
  assert fp[0] == 2. * x[0]              # f_x1 (x0, x1) = 2 * x0
def test_forward_2():
  # start recording a_double operations
  x   = array( [ 2. , 3. ] )
  a_x = independent(x)

  # start recording a2double operations
  u   = array( [ a_x[0] , ad(4) ] )
  a_u = independent(u)

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
# converting integer to double arrays

def test_int2double():
  A   = array([ 
    [ 1, 2, 3 ],
    [ 4, 5, 6 ]
  ])
  x   = array( [ 0, 0, 0 ] )
  a_x = independent(x)
  a_y = numpy.dot(A, a_x)
  f   = adfun(a_x, a_y)
  x   = array( [ 1, 2, 3 ] )
  J   = f.jacobian(x)
  assert numpy.all( A == J )
# jacobian result with type a_double

def test_jacobian_a2():
  print "This test fails and hence has been moved from example/jacobian_a2.py"
  print "to issues.py. The indexing error seems to occur in vec2array.cpp"
  print "where some output statements have been added to help with debugging."
  A   = array([ 
    [ 1., 2., 3. ],
    [ 4., 5., 6. ]
  ])
  x   = array( [ ad(0.), ad(0.), ad(0.) ] )
  a_x = independent(x)
  a_y = numpy.dot(A, a_x)
  f   = adfun(a_x, a_y)
  x   = array( [ ad(1.), ad(2.), ad(3.) ] )
  J   = f.jacobian(x)
  assert numpy.all( A == J )
# jacobian result with type double

def test_jacobian():
  # A = [ 0 1 2 ]  f(x) = A * x
  #     [ 3 4 5 ]
  A   = array([ 
    [ 1., 2., 3. ],
    [ 4., 5., 6. ]
  ])
  x   = array( [ 0., 0., 0. ] )
  a_x = independent(x)
  a_y = numpy.dot(A, a_x)
  f   = adfun(a_x, a_y)
  x   = array( [ 1., 2., 3. ] )
  J   = f.jacobian(x)
  assert numpy.all( A == J )
# First order reverse mode with two levels of taping

def test_reverse_1_a2():
  # start recording a_double operations
  x   = array( [ 2. , 3. ] )
  a_x = independent(x)

  # start recording a2double operations
  u   = array( [ a_x[0] , ad(4) ] )
  a_u = independent(u)

  # stop a2double recording and store operations if f
  a_v = array( [ 2. * a_u[0] * a_u[1] ] )
  f   = adfun(a_u, a_v)              # f(u0, u1) = 2. * u0 * u1

  # evaluate the function f(u) using a_double operations
  u   = array([a_x[0] , 2.*a_x[1]])  # u0 = x0, u1 = 2 * x1
  p   = 0
  fp  = f.forward(p, u)
  assert fp[0] == 2. * u[0] * u[1]

  # derivative of f with respect to u (using a_double)
  p  = 1                         # order of derivative
  w  = array( [ ad(1) ] )        # vector of weights
  fp = f.reverse(p, w)           # derivative of f w.r.t u
  assert fp[0] == 2. * u[1]      # f_u0(u0, u1) = 2. * u1
  assert fp[1] == 2. * u[0]      # f_u1(u0, u1) = 2. * u0

  # stop a_double recording and store operations if g
  a_y = 2. * fp                  # g(x0, x1) = 2 * f_u (x0, 2 * x1)
  g   = adfun(a_x, a_y)          #           = [ 8 * x1 , 4 * x0  ]  

  # evaluate the function g(x) at x = (4, 5) using double operations
  p  = 0
  x  = array( [ 4. , 5. ] )
  gp = g.forward(p, x)
  assert gp[0] == 8. * x[1]
  assert gp[1] == 4. * x[0]

  # derivative of the first component of g with respect to x (using double)
  p  = 1
  w = array( [ 1. , 0. ] )
  gp = g.reverse(p, w)
  assert gp[0] == 0.
  assert gp[1] == 8.
# First order reverse mode.

def test_reverse_1():

  # start record a_double operations
  x   = array( [ 2. , 3. ] )  # value of independent variables
  a_x = independent(x)        # declare independent variables

  # stop recording and store operations in the function object f
  a_y = array( [ 2. * a_x[0] * a_x[1] ] ) # dependent variables
  f   = adfun(a_x, a_y)                   # f(x0, x1) = 2 * x0 * x1

  # evaluate the function at a different argument value
  p  = 0                                 # order zero for function values
  x  = array( [ 3. , 4. ] )              # argument value
  fp = f.forward(p, x)                   # function value
  assert fp[0] == 2. * x[0] * x[1]       # f(x0, x1) = 2 * x0 * x1

  # evalute derivative of f(x0, x1) 
  p  = 1                                 # order one for first derivatives
  w  = array( [ 1. ] )                   # weight in range space
  fp = f.reverse(p, w)                   # derivaitive of weighted function
  assert fp[0] == 2. * x[1]              # f_x0 (x0, x1) = 2 * x1
  assert fp[1] == 2. * x[0]              # f_x1 (x0, x1) = 2 * x0
# the C++ standard math functions
#
# Note that numpy uses the non-standard names arccos, arcsin, and arctan

import math
def test_std_math(): 
   n     = 10
   delta = 10. * numpy.finfo(float).eps
   pi    = numpy.pi
   x     = pi / 6
   a_x   = ad(x)
   a2x   = ad(a_x)

   # all the a_double unary standard math functions
   assert ( arccos(a_x) - math.acos(x) )  < delta
   assert ( arcsin(a_x) - math.asin(x) )  < delta
   assert ( arctan(a_x) - math.atan(x) )  < delta
   assert ( cos(a_x)    - math.cos(x) )   < delta
   assert ( cosh(a_x)   - math.cosh(x) )  < delta
   assert ( exp(a_x)    - math.exp(x) )   < delta
   assert ( log(a_x)    - math.log(x) )   < delta
   assert ( log10(a_x)  - math.log10(x) ) < delta
   assert ( sin(a_x)    - math.sin(x) )   < delta
   assert ( sinh(a_x)   - math.sinh(x) )  < delta
   assert ( sqrt(a_x)   - math.sqrt(x) )  < delta
   assert ( tan(a_x)    - math.tan(x) )   < delta
   assert ( tanh(a_x)   - math.tanh(x) )  < delta

   # all the a2double unary standard math functions
   assert ( arccos(a2x) - math.acos(x) )  < delta
   assert ( arcsin(a2x) - math.asin(x) )  < delta
   assert ( arctan(a2x) - math.atan(x) )  < delta
   assert ( cos(a2x)    - math.cos(x) )   < delta
   assert ( cosh(a2x)   - math.cosh(x) )  < delta
   assert ( exp(a2x)    - math.exp(x) )   < delta
   assert ( log(a2x)    - math.log(x) )   < delta
   assert ( log10(a2x)  - math.log10(x) ) < delta
   assert ( sin(a2x)    - math.sin(x) )   < delta
   assert ( sinh(a2x)   - math.sinh(x) )  < delta
   assert ( sqrt(a2x)   - math.sqrt(x) )  < delta
   assert ( tan(a2x)    - math.tan(x) )   < delta
   assert ( tanh(a_x)   - math.tanh(x) )  < delta

   # example derivative calculation
   x = array( [ x ] )
   a_x = independent(x)
   a_y = sin(a_x)
   f   = adfun(a_x, a_y)
   J   = f.jacobian(x)
   assert ( J[0] - cos(x[0]) ) < delta
