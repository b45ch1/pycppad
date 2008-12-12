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


## testing cos, sin
#x = array([AD_double(pi)])
#Independent(x)
#y = cos(x)
#f = ADFun_double(x, y)
#print y

#p = 0
#xp = array( [ pi ] )
#print f.Forward(p, xp)

#p = 1
#xp = array( [ 1. ] )
#print f.Forward(p,xp)
#print -sin(x)


# testing composite function
N = 10
x = array([AD_double(pi+0.2*i) for i in range(N)])
Independent(x)
y = array(sum(x))
print x
print y
f = ADFun_double(x, y)
#print y

#p = 0
#xp = array( [ pi ] )
#print f.Forward(p, xp)

#p = 1
#xp = array( [ 1. ] )
#print f.Forward(p,xp)




