# $begin independent.py$$ $newlinech #$$
#
# $section independent: Example and Test$$
#
# $index independent, example$$
# $index example, independent$$
#
# $code
# $verbatim%example/independent.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
# Example using a_float ---------------------------------------------------
def test_independent() :
  x   = numpy.array( [ 0., 0., 0. ] )
  a_x = independent(x)    # level 1 independent variables and start recording
  assert type(a_x) == numpy.ndarray
  for j in range(len(x)) :
    assert isinstance(x[j],   float)
    assert isinstance(a_x[j], a_float)
    assert a_x[j] == x[j]
  f   = adfun(a_x, a_x)   # stop level 1 recording
# Example using a2float ---------------------------------------------------
def test_independent_a2() :
  x   = numpy.array( [ 0., 0., 0. ] )
  a_x = independent(x)    # level 1 independent variables and start recording
  a2x = independent(a_x)  # level 2 independent variables and start recording
  assert type(a_x) == numpy.ndarray
  assert type(a2x) == numpy.ndarray
  for j in range(len(x)) :
    assert isinstance(x[j],   float)
    assert isinstance(a_x[j], a_float)
    assert isinstance(a2x[j], a2float)
    assert a_x[j] == x[j]
    assert a2x[j] == x[j]
  a_f = adfun(a2x, a2x)   # stop level 2 recording
  f   = adfun(a_x, a_x)   # stop level 1 recording
# END CODE
