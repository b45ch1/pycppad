# First order forward mode.

def test_forward_1():

  # start record a_double operations
  ad_x = array( [ ad(2) , ad(3) ] )  # value of independent variables
  independent(ad_x)                  # declare independent variables

  # stop recording and store operations in the function object f
  ad_y = array( [ 2. * ad_x[0] * ad_x[1] ] ) # dependent variables
  f = adfun(ad_x, ad_y)                      # f(x0, x1) = 2 * x0 * x1

  # evaluate the function at a different argument value
  p  = 0                                 # order zero for function values
  x  = array( [ 3. , 4. ] )              # argument value
  fp = f.forward(p, x)                   # function value
  assert fp == 2. * x[0] * x[1]          # f(x0, x1) = 2 * x0 * x1

  # evalute partial derivative of f(x0, x1) with respect to x0
  p  = 1                                 # order one for first derivatives
  xp = array( [ 1. , 0. ] )              # direction for differentiation
  fp = f.forward(p, xp)                  # value of directional derivative
  assert fp == 2. * x[1]                 # f_x0 (x0, x1) = 2 * x1

  # evalute partial derivative of f(x0, x1) with respect to x
  p  = 1
  xp = array( [ 0. , 1. ] )              # the x1 direction
  fp = f.forward(p, xp)
  assert fp == 2. * x[0]                 # f_x1 (x0, x1) = 2 * x0
