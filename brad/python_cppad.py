from python_cppad import *
from numpy import array
# ---------------------------------------------------------
print 'Example using ad_double------------------------------------'
level = 1
ad_x = array( [ ad_double(2) , ad_double(3) ] )
independent(ad_x, level)
ad_y = array( [ 2. * ad_x[0] * ad_x[1] ] )
f = adfun_double(ad_x, ad_y)
print 'f(x0, x1) = 2. * x0 * x1'
p  = 0
xp = array( [ 3. , 4. ] )
fp = f.forward(p, xp)
print 'f(3, 4)            = ', fp
p  = 1
xp = array( [ 1. , 0. ] )
fp = f.forward(p, xp)
print 'partial_x0 f(3, 4) = ' , fp
p  = 1
xp = array( [ 0. , 1. ] )
fp = f.forward(p, xp)
print 'partial_x1 f(3, 4) = ' , fp
print 'Example using ad_ad_double------------------------------------'
level = 1
ad_x = array( [ ad_double(2) , ad_double(3) ] )
independent(ad_x, level)
#
level = 2
ad_ad_x = array( [ ad_ad_double(ad_x[0]) , ad_ad_double(ad_x[1]) ] )
independent(ad_ad_x, level)
ad_ad_y = array( [ 2. * ad_ad_x[0] * ad_ad_x[1] ] )
print 'f(x0, x1) = 2. * x0 * x1'
ad_f = adfun_ad_double(ad_ad_x, ad_ad_y)
#
p  = 0
ad_fp = array( [ ad_double(0.) ] )  # kludge to pass back fp
ad_f.forward(p, ad_x, ad_fp)
print 'f(3, 4)            = ', ad_fp
p  = 1
ad_xp = array( [ ad_double(1.) , ad_double(0.) ] )
ad_f.forward(p, ad_xp, ad_fp)
print 'partial_x0 f(3, 4) = ', ad_fp
print 'g(x0, x1) = 2. * partial_x0 f(x0, x1) = 4 * x1'
ad_y = 2. * ad_fp
g = adfun_double(ad_x, ad_y)
p  = 0
xp = array( [ 4. , 5. ] )
gp = g.forward(p, xp)
print 'g(4, 5)            = ', gp
p  = 1
xp = array( [ 1. , 0. ] )
gp = g.forward(p, xp)
print 'partial_x0 g(4, 5) = ', gp
p  = 1
xp = array( [ 0. , 1. ] )
gp = g.forward(p, xp)
print 'partial_x1 g(4, 5) = ', gp
