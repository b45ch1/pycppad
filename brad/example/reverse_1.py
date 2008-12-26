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
