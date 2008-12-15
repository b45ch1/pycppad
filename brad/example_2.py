from cppad import *
print 'Example using ad_ad_double'
# declare level one independent variable vector and start level one recording
level = 1
ad_x = array( [ ad_double(2) , ad_double(3) ] )
independent(ad_x, level)
# declare level two independent variable vector and start level two recording
level = 2
ad_ad_x = array( [ ad_ad_double(ad_x[0]) , ad_ad_double(ad_x[1]) ] )
independent(ad_ad_x, level)
# declare level 2 dependent variable vector and stop level 2 recording
ad_ad_y = array( [ 2. * ad_ad_x[0] * ad_ad_x[1] ] )
ad_f = adfun_ad_double(ad_ad_x, ad_ad_y)
print 'f(x0, x1) = 2. * x0 * x1'
# evaluate the function f(x) using level one independent variable vector
p  = 0
ad_fp = array( [ ad_double(0.) ] )  # kludge to pass back fp
ad_f.forward(p, ad_x, ad_fp)
print 'f(3, 4)            = ', ad_fp
# evaluate the partial of f with respect to the first component
p  = 1
ad_xp = array( [ ad_double(1.) , ad_double(0.) ] )
ad_f.forward(p, ad_xp, ad_fp)
print 'partial_x0 f(3, 4) = ', ad_fp
# declare level 1 dependent variable vector and stop level 1 recording 
ad_y = 2. * ad_fp
g = adfun_double(ad_x, ad_y)
print 'g(x0, x1) = 2. * partial_x0 f(x0, x1) = 4 * x1'
# evaluate the function g(x) at x = (4,5)
p  = 0
xp = array( [ 4. , 5. ] )
gp = g.forward(p, xp)
print 'g(4, 5)            = ', gp
# evaluate the partial of g with respect to the first component
p  = 1
xp = array( [ 1. , 0. ] )
gp = g.forward(p, xp)
print 'partial_x0 g(4, 5) = ', gp
# evaluate the partial of g with respect to the second component
p  = 1
xp = array( [ 0. , 1. ] )
gp = g.forward(p, xp)
print 'partial_x1 g(4, 5) = ', gp
