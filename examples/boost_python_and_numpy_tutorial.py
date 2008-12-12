from boost_python_and_numpy_tutorial import *
from numpy import array

print 'testing double arrays'
y = array([3.,5.],dtype=float)
z = square_elements(y)
print 'y=',y
print 'z=',z

print 'testing ad_double arrays'
ax = array([ad_double(3.5), ad_double(4.)])
ay = my_factorial(ax)
print 'ay=',ay

print 'testing error handling'
x =  array([3.,5.],dtype=float)
y = my_factorial(x)


