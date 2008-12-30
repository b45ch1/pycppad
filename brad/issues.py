# jacobian result with type a_double


from cppad import *

def test_jacobian_a2():
  print "This test fails and hence has been moved from example/jacobian_a2.py"
  print "to issues.py. The indexing error seems to occur in vec2array.cpp"
  print "where some output statements have been added to help with debugging."
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

test_jacobian_a2()
