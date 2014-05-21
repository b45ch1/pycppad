#!/usr/bin/env python
from pycppad import *

def pycppad_test_forward_a2():
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

def pycppad_test_reverse_a2():
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

def pycppad_test_compile_with_debugging() :
  # Cygwin systems have a problem catching exceptions that seems to be a
  # bug in boost-python. We are working on getting this fixed.
  import platform
  uname = (( platform.uname() )[0] )[0:6]
  if not (uname == 'CYGWIN') :
    try :
      x   = numpy.array( [ 1 , 2 ] )
      a_x = independent(x)
      f   = adfun(a_x, a_x)
      x   = numpy.array( [ 1 ] )
      J   = f.jacobian(x)
      # The Line above should raise a CppAD exception because length of x not 2.
      # Currently, CppAD exceptions are returned as ValueError exceptions, but
      # it would be better to have a separate name for them.
      raise RuntimeError
    except ValueError : 
      # exception should come here
      pass

def pycppad_test_compare_op():
  delta = 10. * numpy.finfo(float).eps
  x_array = numpy.array( range(5) )
  y_array = 6. - x_array
  for i in range( len(x_array) ) :
    x   = x_array[i]
    y   = y_array[i]
    a_x = ad(x)
    a_y = ad(y)
    #
    assert (a_x < a_y)  == ( x < y )
    assert (a_x > a_y)  == ( x > y )
    assert (a_x <= a_y) == ( x <= y )
    assert (a_x >= a_y) == ( x >= y )
    assert (a_x == a_y) == ( x == y )
    assert (a_x != a_y) == ( x != y )
  #
  n        = 3.
  x        = numpy.array( [ -2 , +2 ] )
  a_x      = independent(x)
  positive = a_x >= 0 
  # At some level, each element of positive is being converted to a float 
  # before interfacing to pycppad * operator.
  a_y      = ( a_x ** n ) * positive
  f        = adfun(a_x, a_y)
  J        = f.jacobian(x)
  for j in range( len(a_x) ) :
    for i in range( len(a_y) ) :
      if i == j  and x[i] >= 0 :
        assert abs( J[i][j] - n * x[j] ** (n-1) ) < delta
      else :
        assert J[i][j] == 0.
  
  
def pycppad_test_compare_op_a2():
  delta = 10. * numpy.finfo(float).eps
  x_array = numpy.array( range(5) )
  y_array = 6. - x_array
  for i in range( len(x_array) ) :
    x   = x_array[i]
    y   = y_array[i]
    a2x = ad(ad(x))
    a2y = ad(ad(y))
    #
    assert (a2x < a2y)  == ( x < y)
    assert (a2x > a2y)  == ( x > y)
    assert (a2x <= a2y) == ( x <= y)
    assert (a2x >= a2y) == ( x >= y)
    assert (a2x == a2y) == ( x == y)
    assert (a2x != a2y) == ( x != y)
  #
  n        = 3.
  x        = numpy.array( [ -2 , +2 ] )
  a_x      = numpy.array( [ ad(x[0]) , ad(x[1]) ] )
  a2x      = independent(x)
  positive = a2x >= 0 
  # At some level, each element of positive is being converted to a float 
  # before interfacing to pycppad * operator.
  a2y      = ( a2x ** n ) * positive
  f        = adfun(a2x, a2y)
  J        = f.jacobian(x)
  for j in range( len(a2x) ) :
    for i in range( len(a2y) ) :
      if i == j  and x[i] >= 0 :
        assert abs( J[i][j] - n * x[j] ** (n-1) ) < delta
      else :
        assert J[i][j] == 0.

def pycppad_test_ad():
  x = ad(2.)
  y = ad(3.)
  z = ad(2.)
  
  # assert that the conditionals work
  assert x == x
  assert x == z
  assert x != y
  assert x <= x
  assert x <= z
  assert x <= y
  assert x <  y
  
  # assert that conditionals can fail to be true
  assert not x == y
  assert not x != z
  assert not x != x
  assert not x >= y
  assert not x >  y
  
  x = ad(x)
  y = ad(y)
  z = ad(z)
  
  # assert that the conditionals work
  assert x == x
  assert x == z
  assert x != y
  assert x <= x
  assert x <= z
  assert x <= y
  assert x <  y
  
  # assert that conditionals can fail to be true
  assert not x == y
  assert not x != z
  assert not x != x
  assert not x >= y
  assert not x >  y
  
def pycppad_test_array_element_type_is_int():
  A   = numpy.array([ 
    [ 1, 2, 3 ],
    [ 4, 5, 6 ]
  ])
  x   = numpy.array( [ 0, 0, 0 ] )
  a_x = independent(x)
  a_y = numpy.dot(A, a_x)
  f   = adfun(a_x, a_y)
  x   = numpy.array( [ 1, 2, 3 ] )
  J   = f.jacobian(x)
  assert numpy.all( A == J )
  
def pycppad_test_assign_op():
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2., -2., 0.,  4.,   4. ]
  y_list = [ -2,   2,  2., .5,   -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a_y = ad(y)
    #
    tmp = ad(x)
    tmp += a_y
    assert abs( tmp - (x + y) ) < delta
    #
    tmp = ad(x)
    tmp -= y
    assert abs( tmp  - (x - y) ) < delta
    #
    tmp = x
    tmp *= a_y
    assert abs( tmp - x * y ) < delta
    #
    tmp = ad(x)
    tmp /= a_y
    assert abs( tmp - x / y ) < delta
    #
  #
  x   = numpy.array( [ -2 , +2 ] )
  a_x = independent(x)
  a_y = a_x + 2. 
  a_y *= 5.
  f   = adfun(a_x, a_y)
  J   = f.jacobian(x)
  for j in range( len(a_x) ) :
    for i in range( len(a_y) ) :
      if i == j : assert abs( J[i][j] - 5. ) < delta
      else :      assert J[i][j] == 0.

def pycppad_test_assign_op_a2() :
  delta = 10. * numpy.finfo(float).eps
  x_list = [ -2., -2., 0.,  4.,   4. ]
  y_list = [ -2,   2,  2., .5,   -.5 ]  
  for i in range( len(x_list) ) :
    x   = x_list[i]
    y   = y_list[i]
    a2y = ad(ad(y))
    #
    tmp = ad(ad(x))
    tmp += a2y
    assert abs( tmp - (x + y) ) < delta
    #
    tmp = ad(ad(x))
    tmp -= y
    assert abs( tmp  - (x - y) ) < delta
    #
    tmp = x
    tmp *= a2y
    assert abs( tmp - x * y ) < delta
    #
    tmp = ad(ad(x))
    tmp /= a2y
    assert abs( tmp - x / y ) < delta
    #
  #
  a_x   = ad( numpy.array( [ 2 , 2 ] ) )
  a2x = independent(a_x)
  a2y = a2x + 2. 
  a2y *= 5.
  f   = adfun(a2x, a2y)
  J   = f.jacobian(a_x)
  for j in range( len(a2x) ) :
    for i in range( len(a2y) ) :
      if i == j : assert abs( J[i][j] - 5. ) < delta
      else :      assert J[i][j] == 0.
  
def pycppad_test_numeric_op():
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
  x   = numpy.array( [ -2 , +2 ] )
  a_x = independent(x)
  n   = 3
  a_y = a_x ** n
  f   = adfun(a_x, a_y)
  J   = f.jacobian(x)
  for j in range( len(a_x) ) :
    for i in range( len(a_y) ) :
      if i == j : assert abs( J[i][j] - n * x[j] ** (n-1) ) < delta
      else :      assert J[i][j] == 0.

def pycppad_test_numeric_op_a2():
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
  a_x = ad( numpy.array( [ -2 , +2 ] ) )
  a2x = independent(a_x)
  n   = 3.
  a2y = a2x ** n
  a_f = adfun(a2x, a2y)
  J   = a_f.jacobian(a_x)
  for j in range( len(a2x) ) :
    for i in range( len(a2y) ) :
      if i == j : assert abs( J[i][j] - n * a_x[j] ** (n-1) ) < delta
      else :      J[i][j] == 0.
 
def pycppad_test_a_float_variable_info():
  x = ad(2.)
  y = ad(x)
  
  assert x.__str__() == '2'
  assert value(x)     == 2.
  # assert x.id        == 1
  # assert x.taddr     == 0
  
def pycppad_test_ad_a_float_variable_info():
  x = ad(ad(13.0))
  
  assert x.__str__() ==  '13'
  assert value(value(x)) == 13.
  # assert x.id == 1
  # assert x.taddr == 0
  
def pycppad_test_trigonometic_functions():
  N = 5
  x  = numpy.array( [ 2.*n*numpy.pi/N     for n in range(N) ] )
  
  # cos
  ax = independent(x)
  ay = numpy.cos(ax)
  af = adfun(ax, ay)
  J = af.jacobian(x)
  
  assert numpy.sum( abs( numpy.diag( numpy.sin(x)) + J)) == 0
  
  # sin
  ax = independent(x)
  ay = numpy.sin(ax)
  af = adfun(ax, ay)
  J = af.jacobian(x)
  assert numpy.prod( numpy.diag( numpy.cos(x)) == J)


def pycppad_test_pow():
  x   = numpy.array( [ 3, 2] )
  ax  = numpy.array( [ ad(x[0]), ad(x[1])] )
  ay = numpy.array([ ax[0]**2, ax[1]**2, ax[0]**2., ax[1]**2., ax[0]**0.5, ax[1]**0.5, ax[0]**ad(2), ax[1]**ad(2)])
  y = numpy.array([ x[0]**2, x[1]**2, x[0]**2., x[1]**2., x[0]**0.5, x[1]**0.5, x[0]**2, x[1]**2])
  
  assert numpy.prod(ay == y)
  

def pycppad_test_multi_level_taping_and_higher_order_forward_derivatives():
  ok = True
  level = 1
  x = numpy.array( [ 2 , 3 ] )
  ad_x = independent(x)
  # declare level two independent variable vector and start level two recording
  level = 2
  ad_ad_x = independent(ad_x)
  # declare level 2 dependent variable vector and stop level 2 recording
  ad_ad_y = numpy.array( [ 2. * ad_ad_x[0] * ad_ad_x[1] ] )
  ad_f = adfun(ad_ad_x, ad_ad_y) # f(x0, x1) = 2. * x0 * x1
  # evaluate the function f(x) using level one independent variable vector
  p  = 0
  ad_fp = ad_f.forward(p, ad_x)
  ok = ok and (ad_fp == 2. * ad_x[0] * ad_x[1])
  # evaluate the partial of f with respect to the first component
  p  = 1
  ad_xp = numpy.array( [ ad(1.) , ad(0.) ] )
  ad_fp = ad_f.forward(p, ad_xp)
  ok = ok and (ad_fp == 2. * ad_x[1])
  # declare level 1 dependent variable vector and stop level 1 recording 
  ad_y = 2. * ad_fp
  g = adfun(ad_x, ad_y) # g(x0, x1) = 2. * partial_x0 f(x0, x1) = 4 * x1
  # evaluate the function g(x) at x = (4,5)
  p  = 0
  x  = numpy.array( [ 4. , 5. ] )
  gp = g.forward(p, x)
  ok = ok and (gp == 4. * x[1])
  # evaluate the partial of g with respect to x0
  p  = 1
  xp = numpy.array( [ 1. , 0. ] )
  gp = g.forward(p, xp)
  ok = ok and (gp == 0.)
  # evaluate the partial of g with respect to x1
  p  = 1
  xp = numpy.array( [ 0. , 1. ] )
  gp = g.forward(p, xp)
  ok = ok and (gp == 4.)
  
  assert ok

def pycppad_test_multi_level_taping_and_higher_order_reverse_derivatives():

  # domain space vector
  x = numpy.array([0., 1.])

  # declare independent variables and start recording
  ax = independent(x);

  ay = numpy.array([ax[0] * ax[0] * ax[1]])

  # create f : X -> Y and stop recording
  af = adfun (ax, ay);

  # use first order reverse mode to evaluate derivative of y[0]
  # and use the values in X for the independent variables.
  w = numpy.zeros(1)
  w[0] = 1.

  y = af.forward(0, numpy.array([0.,1.]))
  dw = af.reverse(1, w);
  assert dw[0] == 2.*ax[0]*ax[1]
  assert dw[1] == ax[0]*ax[0]

  # use zero order forward mode to evaluate y at x = (3, 4)
  # and use the template parameter Vector for the vector type

  x =  numpy.array([3.,4.])
  y = af.forward(0,x)
  assert y[0] == x[0]*x[0]*x[1]

  # use first order reverse mode to evaluate derivative of y[0]
  # and using the values in x for the independent variables.
  w[0] = 1.
  dw   = af.reverse(1, w)
  
  assert dw[0] ==  2.*x[0]*x[1]
  assert dw[1] ==   x[0]*x[0]
  
def pycppad_test_jacobian():
  N = 4
  A = numpy.array([n+1. for n in range(N*N)]).reshape((N,N))
  def f(x):
    return numpy.dot(A,x)
  
  x = numpy.array([0. for n in range(N) ])
  ax = independent(x)
  ay = f(ax)
  af = adfun (ax, ay);
  x = numpy.array([1. for n in range(N)])
  
  J = af.jacobian(x)
  
  assert numpy.prod( A == J )
  
def pycppad_test_hessian():
  N = 4
  A = numpy.ones((N,N)) + 2.*numpy.eye(N)
  def fun(x):
    return numpy.array([0.5* numpy.dot(x,numpy.dot(A,x))])
  
  x = numpy.array([0. for n in range(N) ])
  a_x = independent(x)
  a_y = fun(a_x)
  f   = adfun (a_x, a_y);
  x = numpy.array([1. for n in range(N)])
  w = numpy.array( [ 1. ] )
  H = f.hessian(x, w)
  assert numpy.prod( A == H )

def pycppad_test_mixed_element_types():
	x   = numpy.array( [ 1 , 2. ], dtype=object )
	ok  = False
	try :
		a_x = independent(x)
	except NotImplementedError :
		x[0] = 0.
		a_x  = independent(x)
		ok   = True
	assert ok
	#
	a_y = numpy.array( [ ad(0) , 1. ], dtype=object )
	ok  = False
	try :
		f = adfun(a_x, a_y)
	except NotImplementedError :
		a_y[1] = ad(1)
		f      = adfun(a_x, a_y) 
		ok     = True
	assert ok
	#

import sys
if __name__ == "__main__" :
  import sys
  ok = len(sys.argv) == 2
  if ok :
     if sys.argv[1] == 'True' :
         with_debugging = True
     elif sys.argv[1] == 'False' :
         with_debugging = False
     else :
         ok = False;
  if not ok :
    print 'usage: python test_more.py with_debugging'
    print '       where with_debugging is either True or False'
    sys.exit(1)
  #
  number_ok   = 0
  number_fail = 0
  list_of_globals = sorted( globals().copy() )
  for g in list_of_globals :
    if g == 'pycppad_test_compile_with_debugging' and (not with_debugging) :
      pass
    elif g[:13] == "pycppad_test_" :
      ok = True
      try :
        eval("%s()" % g)
      except AssertionError :
        ok = False
      if ok : 
        print "OK:    %s" % g[13:]
        number_ok = number_ok + 1
      else : 
        print "Error: %s" % g[13:]
        number_fail = number_fail + 1
  if number_fail == 0 : 
    print "All %d tests passed" % number_ok
    sys.exit(0)
  else :
    print "%d tests failed" % number_fail 
    sys.exit(1)
