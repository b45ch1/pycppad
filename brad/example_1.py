from cppad import *
print 'Example using ad_double'
# declare independent variable vector and start recording
level = 1
ad_x = array( [ ad_double(2) , ad_double(3) ] )
independent(ad_x, level)
# declare dependent variable vector and stop recording
ad_y = array( [ 2. * ad_x[0] * ad_x[1] ] )
f = adfun_double(ad_x, ad_y)
print 'f(x0, x1) = 2. * x0 * x1'
# evaluate the function at a different argument value
p  = 0
xp = array( [ 3. , 4. ] )
fp = f.forward(p, xp)
print 'f(3, 4)            = ', fp
# evalute partial derivative with respect to first component
p  = 1
xp = array( [ 1. , 0. ] )
fp = f.forward(p, xp)
print 'partial_x0 f(3, 4) = ' , fp
# evalute partial derivative with respect to second component
p  = 1
xp = array( [ 0. , 1. ] )
fp = f.forward(p, xp)
print 'partial_x1 f(3, 4) = ' , fp
