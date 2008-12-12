#!/usr/bin/env python
import sys
sys.path = ['./release'] + sys.path

from python_cppad import *
from numpy import array
x = array( [ AD_double(2) , AD_double(3) ] )
independent(x);
y = array( [ x[0] * x[1] ] );
f = ADFun_double(x, y)
print 'f(x0, x1) = x0 * x1'
p  = 0
xp = array( [ 3. , 4. ] )
fp = f.Forward(p, xp)
print 'f(3, 4)            = ', fp
p  = 1
xp = array( [ 1. , 0. ] )
fp = f.Forward(p, xp)
print 'partial_x0 f(3, 4) = ' , fp
p  = 1
xp = array( [ 0. , 1. ] )
fp = f.Forward(p, xp)
print 'partial_x1 f(3, 4) = ' , fp
print 'Begin: example python_cppad error message'
p  = 1
xp = array( [ 0. , 1., 0. ] )
fp = f.Forward(p, xp)