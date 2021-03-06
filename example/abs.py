# $begin abs.py$$ $newlinech #$$
#
# $section abs: Example and Test$$
#
# $index abs, example$$
# $index example, abs$$
#
# $code
# $verbatim%example/abs.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
# Example using a_float ----------------------------------------------------
from pycppad import *
def pycppad_test_abs() :
  x   = numpy.array( [ -1.,  0.,  1.] )
  n   = len(x)
  a_x = independent(x)
  a_y = abs( a_x )
  f   = adfun(a_x, a_y)
  f.forward(0, x)
  dx  = numpy.zeros(n, dtype=float)
  for i in range( n ) :
    dx[i] = 1.
    df    = f.forward(1, dx)
    if x[i] > 0. :
      assert df[i] == +1.
    elif x[i] < 0. :
      assert df[i] == -1.
    else :
      # There was a change in the CppAD specifictions for the abs function
      # see 12-30 on http://www.coin-or.org/CppAD/Doc/whats_new_11.htm
      assert df[i] == +1. or df[i] == 0.
    dx[i] = -1.
    df    = f.forward(1, dx)
    if x[i] > 0. :
      assert df[i] == -1.
    elif x[i] < 0. :
      assert df[i] == +1.
    else :
      assert df[i] == +1 or df[i] == 0.
    dx[i] = 0.
# Example using a2float ----------------------------------------------------
def pycppad_test_abs_a2() :
  x   = ad( numpy.array( [-1,  0,  1] ) )
  n   = len(x)
  a_x = independent(x)
  a_y = abs( a_x )
  f   = adfun(a_x, a_y)
  f.forward(0, x)
  dx  = numpy.array( list( ad(0) for i in range(n) ) )
  for i in range( n ) :
    dx[i] = ad(0)
  for i in range( n ) :
    dx[i] = ad(1)
    df    = f.forward(1, dx)
    if x[i] > 0. :
      assert df[i] == +1.
    elif x[i] < 0. :
      assert df[i] == -1.
    else :
      assert df[i] == +1. or df[i] == 0.
    dx[i] = ad(-1)
    df    = f.forward(1, dx)
    if x[i] > 0. :
      assert df[i] == -1.
    elif x[i] < 0. :
      assert df[i] == +1.
    else:
      assert df[i] == +1. or df[i] == 0.
    dx[i] = ad(0)
# END CODE
