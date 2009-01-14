# jacobian result with type a_float

from cppad import *
def test_jacobian_a2():
  A   = array([ 
    [ 1., 2., 3. ],
    [ 4., 5., 6. ]
  ])
  x   = array( [ ad(0.), ad(0.), ad(0.) ] )
  a_x = independent(x)
  a_y = numpy.dot(A, a_x)
  f   = adfun(a_x, a_y)
  x   = array( [ ad(1.), ad(2.), ad(3.) ] )
  J   = f.jacobian(x)
  assert numpy.all( A == J )
