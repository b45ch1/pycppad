import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH

from cppad import *


def test_conditionals():
	x = adouble(2.)
	y = adouble(3.)
	z = adouble(2.)
	
	# assert that the conditionals work
	assert x == x
	assert x == z
	assert x != y
	assert x <= x
	assert x <= z
	assert x <= y
	assert x <  y
	
	# assert that conditionals can fail to be true
	assert not x == y
	assert not x != z
	assert not x != x
	assert not x >= y
	assert not x >  y
	
	
def test_ad_adouble():
	x2 = addouble(ad(3.))
	y2 = addouble(ad(3.))
	z2 = addouble(ad(4.))

	assert False
	
def test_elementary_ad_operations():
	x = adouble(2.)
	y = adouble(3.)
	
	assert x + y == adouble(5.)
	assert x - y == adouble(-1.)
	assert x * y == adouble(6.)
	assert x / y == adouble(2./3.)
	
	x += y
	assert x == adouble(5.)
	x -= y
	assert x == adouble(2.)
	
	
	
	
	
	
#def test_adouble_variable_info():
	#x = ad(2.)
	#y = addouble(x)
	#print x
	#print y
	#assert False
	#print x
	#print x.value
	#print x.id
	#print x.taddr
	
	#assert x.__str__() == str(2.)
	#assert x.value     == 2.
	#assert x.id        == 1
	#assert x.taddr     == 0
	
#def test_ad_adouble_variable_info():
	#x = ad(ad(13.0))
	#print x
	#print x.value
	#print x.id
	#print x.taddr
	
	#assert x.__str__() ==  '13'
	#assert x.value == 13.
	#assert x.id == 1
	#assert x.taddr == 0	
	

#def test_adouble():
	#x = ad(3.)
	#y = ad(3.)
	#z = ad(4.)
	
	#assert x == y
	#assert not x == z


	
#def test_deprecated_multi_level_taping_and_higher_order_derivatives():
	#ok = True
	#level = 1
	#ad_x = numpy.array( [ ad_double(2) , ad_double(3) ] )
	#independent(ad_x)
	## declare level two independent variable vector and start level two recording
	#level = 2
	#ad_ad_x = numpy.array( [ ad_ad_double(ad_x[0]) , ad_ad_double(ad_x[1]) ] )
	#independent(ad_ad_x)
	## declare level 2 dependent variable vector and stop level 2 recording
	#ad_ad_y = numpy.array( [ 2. * ad_ad_x[0] * ad_ad_x[1] ] )
	#ad_f = adfun_ad_double(ad_ad_x, ad_ad_y) # f(x0, x1) = 2. * x0 * x1
	## evaluate the function f(x) using level one independent variable vector
	#p  = 0
	#ad_fp = ad_f.Forward(p, ad_x)
	#ok = ok and (ad_fp == 2. * ad_x[0] * ad_x[1])
	## evaluate the partial of f with respect to the first component
	#p  = 1
	#ad_xp = numpy.array( [ ad_double(1.) , ad_double(0.) ] )
	#ad_fp = ad_f.Forward(p, ad_xp)
	#ok = ok and (ad_fp == 2. * ad_x[1])
	## declare level 1 dependent variable vector and stop level 1 recording 
	#ad_y = 2. * ad_fp
	#g = adfun_double(ad_x, ad_y) # g(x0, x1) = 2. * partial_x0 f(x0, x1) = 4 * x1
	## evaluate the function g(x) at x = (4,5)
	#p  = 0
	#x  = numpy.array( [ 4. , 5. ] )
	#gp = g.Forward(p, x)
	#ok = ok and (gp == 4. * x[1])
	## evaluate the partial of g with respect to x0
	#p  = 1
	#xp = numpy.array( [ 1. , 0. ] )
	#gp = g.Forward(p, xp)
	#ok = ok and (gp == 0.)
	## evaluate the partial of g with respect to x1
	#p  = 1
	#xp = numpy.array( [ 0. , 1. ] )
	#gp = g.Forward(p, xp)
	#ok = ok and (gp == 4.)
	
	#assert ok

#def test_multi_level_taping_and_higher_order_derivatives():
	#ok = True
	#level = 1
	#ad_x = numpy.array( [ ad(2) , ad(3) ] )
	#print ad_x
	#independent(ad_x)
	## declare level two independent variable vector and start level two recording
	#level = 2
	#ad_ad_x = numpy.array( [ ad(ad_x[0]) , ad(ad_x[1]) ] )
	#print ad_ad_x
	#independent(ad_ad_x)
	## declare level 2 dependent variable vector and stop level 2 recording
	#ad_ad_y = numpy.array( [ 2. * ad_ad_x[0] * ad_ad_x[1] ] )
	#print ad_ad_y
	##ad_f = adfun_ad_double(ad_ad_x, ad_ad_y) # f(x0, x1) = 2. * x0 * x1
	### evaluate the function f(x) using level one independent variable vector
	##p  = 0
	##ad_fp = ad_f.Forward(p, ad_x)
	##ok = ok and (ad_fp == 2. * ad_x[0] * ad_x[1])
	### evaluate the partial of f with respect to the first component
	##p  = 1
	##ad_xp = numpy.array( [ ad(1.) , ad(0.) ] )
	##ad_fp = ad_f.Forward(p, ad_xp)
	##ok = ok and (ad_fp == 2. * ad_x[1])
	### declare level 1 dependent variable vector and stop level 1 recording 
	##ad_y = 2. * ad_fp
	##g = adfun_double(ad_x, ad_y) # g(x0, x1) = 2. * partial_x0 f(x0, x1) = 4 * x1
	### evaluate the function g(x) at x = (4,5)
	##p  = 0
	##x  = numpy.array( [ 4. , 5. ] )
	##gp = g.Forward(p, x)
	##ok = ok and (gp == 4. * x[1])
	### evaluate the partial of g with respect to x0
	##p  = 1
	##xp = numpy.array( [ 1. , 0. ] )
	##gp = g.Forward(p, xp)
	##ok = ok and (gp == 0.)
	### evaluate the partial of g with respect to x1
	##p  = 1
	##xp = numpy.array( [ 0. , 1. ] )
	##gp = g.Forward(p, xp)
	##ok = ok and (gp == 4.)
	#assert False
	#assert ok





