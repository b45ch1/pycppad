#!/usr/bin/env python
import sys
sys.path = ['.'] + sys.path

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
x = array([ad_double(pi+0.2*i) for i in range(N)])
independent(x)
y = array([sum(x)])
print x
print y
f = adfun_double(x, y)
print y

p = 0
xp = array( [ 0.2*i for i in range(N) ] )
print f.Forward(p, xp)

p = 1
xp = array( [ 0.3*i for i in range(N) ] )
print f.Forward(p,xp)



# TESTING MULTILEVEL TAPING AND HIGHER ORDER DERIVATIVES
ok = True
level = 1
ad_x = array( [ ad_double(2) , ad_double(3) ] )
independent(ad_x)
# declare level two independent variable vector and start level two recording
level = 2
ad_ad_x = array( [ ad_ad_double(ad_x[0]) , ad_ad_double(ad_x[1]) ] )
independent(ad_ad_x)
# declare level 2 dependent variable vector and stop level 2 recording
ad_ad_y = array( [ 2. * ad_ad_x[0] * ad_ad_x[1] ] )
ad_f = adfun_ad_double(ad_ad_x, ad_ad_y) # f(x0, x1) = 2. * x0 * x1
# evaluate the function f(x) using level one independent variable vector
p  = 0
ad_fp = ad_f.Forward(p, ad_x)
ok = ok and (ad_fp == 2. * ad_x[0] * ad_x[1])
# evaluate the partial of f with respect to the first component
p  = 1
ad_xp = array( [ ad_double(1.) , ad_double(0.) ] )
ad_fp = ad_f.Forward(p, ad_xp)
ok = ok and (ad_fp == 2. * ad_x[1])
# declare level 1 dependent variable vector and stop level 1 recording 
ad_y = 2. * ad_fp
g = adfun_double(ad_x, ad_y) # g(x0, x1) = 2. * partial_x0 f(x0, x1) = 4 * x1
# evaluate the function g(x) at x = (4,5)
p  = 0
x  = array( [ 4. , 5. ] )
gp = g.Forward(p, x)
ok = ok and (gp == 4. * x[1])
# evaluate the partial of g with respect to x0
p  = 1
xp = array( [ 1. , 0. ] )
gp = g.Forward(p, xp)
ok = ok and (gp == 0.)
# evaluate the partial of g with respect to x1
p  = 1
xp = array( [ 0. , 1. ] )
gp = g.Forward(p, xp)
ok = ok and (gp == 4.)

print ok



