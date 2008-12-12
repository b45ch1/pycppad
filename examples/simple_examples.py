#!/usr/bin/env python
import sys
sys.path = ['./release'] + sys.path

from cppad import *
from numpy import *
#from numpy import array,pi,cos


## testing the multiplication 
#x = array( [ AD_double(2) , AD_double(3) ] )
#independent(x);
#y = array( [ x[0] * x[1] ] )
#f = ADFun_double(x, y)
#print 'f(x0, x1) = x0 * x1'
#p  = 0
#xp = array( [ 3. , 4. ] )
#fp = f.Forward(p, xp)
#print 'f(3, 4)            = ', fp
#p  = 1
#xp = array( [ 1. , 0. ] )
#fp = f.Forward(p, xp)
#print 'partial_x0 f(3, 4) = ' , fp
#p  = 1
#xp = array( [ 0. , 1. ] )
#fp = f.Forward(p, xp)
#print 'partial_x1 f(3, 4) = ' , fp
#print 'Begin: example python_cppad error message'
#p  = 1
#xp = array( [ 0. , 1., 0. ] )
#fp = f.Forward(p, xp)


# testing cos
x = array([AD_double(pi)])
Independent(x)
y = cos(x)
f = ADFun_double(x, y)

p = 0
xp = array( [ 1.  ] )
print f.Forward(p, xp)





