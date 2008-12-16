import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH

from cppad import *


def test_conditionals():
	x = ad_double(2.)
	y = ad_double(3.)
	z = ad_double(2.)
	
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

def test_the_different_methods_to_use_ad_double():
	x = ad(3.)
	y = ad(4.)
	
	z = x * y
	
	print z
	
	#y = ad(5.)
	#u = ad(ad(2.))
	#v = ad(ad(4.))
	
	#z = x * y

	#print x
	#print y
	
	assert False
	




def test_elementary_ad_operations():
	x = ad_double(2.)
	y = ad_double(3.)
	
	assert x + y == ad_double(5.)
	assert x - y == ad_double(-1.)
	assert x * y == ad_double(6.)
	assert x / y == ad_double(2./3.)
	
	x += y
	assert x == ad_double(5.)
	x -= y
	assert x == ad_double(2.)
