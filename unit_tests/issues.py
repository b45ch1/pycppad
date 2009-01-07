import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH

from cppad import *
import numpy

def test_int_pow_a_double() :
	x = 1
	y = ad(1)
	assert x**y == 1
	
	
# The code above violates the specifications for CppAD. If it is run
# with debugging, CppAD generates an error message. If run without debugging
# CppAD assumes that its specifications have been met.
#def test_independent_exception_handling():
#	""" this results in a segmentation fault if run without debugging"""
#	a_x = numpy.array( [ad(1.)] )
#	independent(a_x)
#	a_y = a_x
#	f = adfun(a_x, a_y)
