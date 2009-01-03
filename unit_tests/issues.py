import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH

from cppad import *
import numpy


def test_numpy_array_data_type() :
	my_list  = [1 , 2 ]
	my_array = numpy.asarray(my_list, dtype=float)
	assert type( my_array[0] ) == float
	
	
def test_independent_exception_handling():
	""" this results in a segmentation fault"""
	ax = numpy.array([ad(1.)])
	independent(ax)
	a_y = ax
	f = adfun(ax, a_y)
	#y = f.forward(0, x) 
	#assert y[0] == 1
