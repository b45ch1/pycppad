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
from cppad import *
def test_independent() :
  x   = array( [ 0., 0., 0. ] )
  a_x = independent(x)    # declare independent variables and start recording
  assert type(a_x) == numpy.ndarray
  for j in range(len(x)) :
    assert isinstance(x[j],   float)
    assert isinstance(a_x[j], a_float)
    assert a_x[j] == x[j]
  f   = adfun(a_x, a_x)   # stop recording
# END CODE
