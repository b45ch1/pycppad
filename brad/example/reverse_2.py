# $begin reverse_2.py$$ $newlinech #$$
#
# $section Reverse Order Two: Example and Test$$
#
# $index reverse, order two example$$
# $index order, reverse two example$$
# $index two, order reverse example$$
# $index example, order two reverse$$
#
# $code
# $verbatim%example/reverse_2.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pyad import *
# Example using a_float ------------------------------------------------------
def test_reverse_2():

  # start record a_float operations
  x   = numpy.array( [ 2. , 3. ] )  # value of independent variables
  a_x = independent(x)              # declare a_float independent variables

  # stop recording and store operations in the function object f
  a_y = numpy.array( [ 2. * a_x[0] * a_x[1] ] ) # dependent variables
  f   = adfun(a_x, a_y)                         # f(x0, x1) = 2 * x0 * x1

  # evaluate the function at same argument value
  p   = 0                           # derivative order
  x_p = x                           # zero order Taylor coefficient
  f_p = f.forward(0, x_p)           # function value
  assert f_p[0] == 2. * x[0] * x[1] # f(x0, x1) = 2 * x0 * x1

  # evalute partial derivative with respect to x[0]
  p  = 1                            # derivative order
  x_p = numpy.array( [ 1. , 0 ] )   # first order Taylor coefficient
  f_p = f.forward(1, x_p)           # partial w.r.t. x0
  assert f_p[0] == 2. * x[1]        # f_x0 (x0, x1) = 2 * x1
 
  # evaluate derivative of partial w.r.t. x[0]
  p  = 2                            # derivative order
  w  = numpy.array( [1.] )          # weighting vector
  dw = f.reverse(p, w)              # derivaitive of weighted function
  assert dw[0] == 0.                # f_x0_x1 (x0, x1) = 0 
  assert dw[1] == 2.                # f_x0_x1 (x0, x1) = 2 

# Example using a2float ------------------------------------------------------
def test_reverse_2_a2():

  # start record a_float operations
  x   = numpy.array( [ 2. , 3. ] )  # value of independent variables
  a_x = ad(x)                       # value of independent variables
  a2x = independent(a_x)            # declare a2float independent variables

  # stop recording and store operations in the function object f
  a2y = numpy.array( [ 2. * a2x[0] * a2x[1] ] ) # dependent variables
  a_f = adfun(a2x, a2y)                         # f(x0, x1) = 2 * x0 * x1

  # evaluate the function at same argument value
  p   = 0                           # derivative order
  x_p = a_x                         # zero order Taylor coefficient
  f_p = a_f.forward(0, x_p)         # function value
  assert f_p[0] == 2. * x[0] * x[1] # f(x0, x1) = 2 * x0 * x1

  # evalute partial derivative with respect to x[0]
  p  = 1                            # derivative order
  x_p = ad(numpy.array([1. , 0 ]))  # first order Taylor coefficient
  f_p = a_f.forward(1, x_p)         # partial w.r.t. x0
  assert f_p[0] == 2. * x[1]        # f_x0 (x0, x1) = 2 * x1
 
  # evaluate derivative of partial w.r.t. x[0]
  p  = 2                            # derivative order
  w  = ad(numpy.array( [1.] ))      # weighting vector
  dw = a_f.reverse(p, w)            # derivaitive of weighted function
  assert dw[0] == 0.                # f_x0_x1 (x0, x1) = 0 
  assert dw[1] == 2.                # f_x0_x1 (x0, x1) = 2 

# END CODE
