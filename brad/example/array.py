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
  y   = array([ 3 , 2 , 1 ])
  assert type(y) == numpy.ndarray and len(y) == 3
  assert isinstance(y[0], float) and y[0] == 3.
  y   = array([ 3. , 2. , 1. ])
  assert type(y) == numpy.ndarray and len(y) == 3
  assert isinstance(y[2], float) and y[1] == 2.
  y   = array( [ ad(3), ad(2) , ad(1) ] )
  assert type(y) == numpy.ndarray and len(y) == 3
  assert isinstance(y[0], a_float) and y[2] == 1.
  y   = array( [ ad(ad(3)), ad(ad(2)) , ad(ad(1)) ] )
  assert type(y) == numpy.ndarray and len(y) == 3
  assert isinstance(y[0], a2float) and y[0] == 3.
# END CODE
