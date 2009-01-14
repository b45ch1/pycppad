# converting integer to double arrays

from cppad import *
def test_int2double():
  A   = numpy.array([ 
    [ 1, 2, 3 ],
    [ 4, 5, 6 ]
  ])
  x   = numpy.array( [ 0, 0, 0 ] )
  a_x = independent(x)
  a_y = numpy.dot(A, a_x)
  f   = adfun(a_x, a_y)
  x   = numpy.array( [ 1, 2, 3 ] )
  J   = f.jacobian(x)
  assert numpy.all( A == J )
