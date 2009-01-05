# $begin array.py$$ $newlinech #$$
# $spell
#	Numpy
# $$
#
# $section array: Example and Test$$
#
# $index array, example$$
# $index example, array$$
#
# $code
# $verbatim%example/array.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
def test_array() :
  x   = array( [ 3 , 2 , 1 ] )
  assert type(x) == numpy.ndarray
  assert isinstance(x[0], float) and len(x) == 3 and x[0] == 3.
  x   = array( [ 3. , 2. , 1. ] )
  assert type(x) == numpy.ndarray
  assert isinstance(x[0], float) and len(x) == 3 and x[1] == 2.
  x   = array( [ ad(3), ad(2) , ad(1) ] )
  assert type(x) == numpy.ndarray
  assert isinstance(x[0], a_float) and len(x) == 3 and x[2] == 1.
  x   = array( [ ad(ad(3)), ad(ad(2)) , ad(ad(1)) ] )
  assert type(x) == numpy.ndarray
  assert isinstance(x[0], a2float) and len(x) == 3 and x[0] == 3.
# END CODE
