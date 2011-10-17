# $begin optimize.py$$ $newlinech #$$
# $spell
# $$
#
# $section Optimize Function Object: Example and Test$$
#
# $index optimize, example$$
# $index operation, optimize sequence$$
# $index sequence, optimize operation$$
# $index example, optimize$$
#
# $code
# $verbatim%example/optimize.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
import time
# Example using a_float -----------------------------------------------------
def pycppad_test_optimize():
  # create function with many variables that are get removed by optimize
  n_sum = 10000
  x     = numpy.array( [ 0. ] )
  a_x   = independent(x)
  a_sum = 0.
  for i in range(n_sum) :
    a_sum = a_sum + a_x[0];
  a_y = numpy.array( [ a_sum ] )
  f   = adfun(a_x, a_y)
  # time for a forward operations before optimize
  x          = numpy.array( [ 1. ] )
  t0         = time.time()
  sum_before = f.forward(0, x)
  sec_before = time.time() - t0
  # time for a forward operations after optimize
  f.optimize()
  t0         = time.time()
  sum_after  = f.forward(0, x)
  sec_after  = time.time() - t0
  assert sum_before == float(n_sum)
  assert sum_after  == float(n_sum)
  # expect sec_before to be less than 2 times sec_after
  assert( sec_after * 1.5 <= sec_before )
# END CODE
