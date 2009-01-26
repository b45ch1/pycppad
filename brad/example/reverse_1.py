# $begin reverse_1.py$$ $newlinech #$$
#
# $section Reverse Order One: Example and Test$$
#
# $index reverse, order one example$$
# $index order, reverse one example$$
# $index one, order reverse example$$
# $index example, order one reverse$$
#
# $code
# $verbatim%example/reverse_1.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pyad import *
# Example using a_float ------------------------------------------------------
def test_reverse_1():

  # start record a_float operations
  x   = numpy.array( [ 2. , 3. ] )  # value of independent variables
  a_x = independent(x)              # declare a_float independent variables

  # stop recording and store operations in the function object f
  a_y = numpy.array( [ 2. * a_x[0] * a_x[1] ] ) # dependent variables
  f   = adfun(a_x, a_y)                         # f(x0, x1) = 2 * x0 * x1

  # evaluate the function at a different argument value
  p  = 0                                 # order zero for function values
  x  = numpy.array( [ 3. , 4. ] )        # argument value
  fp = f.forward(p, x)                   # function value
  assert fp[0] == 2. * x[0] * x[1]       # f(x0, x1) = 2 * x0 * x1

  # evalute derivative of f(x0, x1) 
  p  = 1                                 # order one for first derivatives
  w  = numpy.array( [ 1. ] )             # weight in range space
  fp = f.reverse(p, w)                   # derivaitive of weighted function
  assert fp[0] == 2. * x[1]              # f_x0 (x0, x1) = 2 * x1
  assert fp[1] == 2. * x[0]              # f_x1 (x0, x1) = 2 * x0

# Example using a2float ------------------------------------------------------
def test_reverse_1_a2():

  # start record a_float operations
  a_x = ad(numpy.array( [ 2. , 3. ] ))  # value of independent variables
  a2x = independent(a_x)                # declare a2float independent variables

  # stop recording and store operations in the function object f
  a2y = numpy.array( [ 2. * a2x[0] * a2x[1] ] ) # dependent variables
  a_f = adfun(a2x, a2y)                         # f(x0, x1) = 2 * x0 * x1

  # evaluate the function at a different argument value
  p   = 0                                 # order zero for function values
  a_x = ad(numpy.array( [ 3. , 4. ] ))    # argument value
  a_fp = a_f.forward(p, a_x)              # function value
  assert a_fp[0] == 2. * a_x[0] * a_x[1]  # f(x0, x1) = 2 * x0 * x1

  # evalute derivative of f(x0, x1) 
  p  = 1                                 # order one for first derivatives
  a_w  = ad(numpy.array( [ 1. ] ))       # weight in range space
  a_fp = a_f.reverse(p, a_w)             # derivaitive of weighted function
  assert a_fp[0] == 2. * a_x[1]          # f_x0 (x0, x1) = 2 * x1
  assert a_fp[1] == 2. * a_x[0]          # f_x1 (x0, x1) = 2 * x0
# END CODE
