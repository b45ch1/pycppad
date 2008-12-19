from cppad import *
ok      = True
# declare independent variable vector and start recording
ad_x = array( [ ad_double(2) , ad_double(3) ] )
independent(ad_x)
# declare dependent variable vector and stop recording
ad_y = array( [ 2. * ad_x[0] * ad_x[1] ] )
f = adfun_double(ad_x, ad_y) # f(x0, x1) = 2. * x0 * x1
# evaluate the function at a different argument value
p  = 0
x  = array( [ 3. , 4. ] )
fp = f.forward(p, x)
ok = ok and (fp == 2. * x[0] * x[1])
# evalute partial derivative with respect to x0
p  = 1
xp = array( [ 1. , 0. ] )
fp = f.forward(p, xp)
ok = ok and (fp == 2. * x[1])
# evalute partial derivative with respect to x1
p  = 1
xp = array( [ 0. , 1. ] )
fp = f.forward(p, xp)
ok = ok and (fp == 2. * x[0])
if( ok ) :
	print 'OK:    example_1: getting started.'
else :
	print 'Error: example_1: getting started.'
