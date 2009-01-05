import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH

from cppad import *
import numpy

# Comment out these test because 
# segmentation fault prevents proper error reporting
#
#def test_numpy_array_data_type() :
#	my_list  = [1 , 2 ]
#	my_array = numpy.asarray(my_list, dtype=float)
#	
#	# this works, but type( my_array[0] ) != float
#	assert isinstance(my_array[0], float)
#
#def test_int_pow_a_double() :
#	x = 1
#	y = ad(1)
#	assert x**y == 1
	
	
def test_independent_exception_handling():
	""" this results in a segmentation fault"""
	ax = numpy.array([ad(1.)])
	independent(ax)
	f = adfun(ax, ax)
