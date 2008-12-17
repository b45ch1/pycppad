"""\
This is PyCppAD, a Python module to differentiate complex algorithms written in Python.
PyCppAD is a wrapper to CppAD (C++).

Example:
	x = numpy.array([ad(2.), ad(3.)])
	y = numpy.array([x[0]*x[1]])
	independent(x)
	adf = adfun(x,y)

	f = adf.forward(0, numpy.array([7.,13.]))
	g = adf.forward(1, numpy.array([1., 0.]))

"""

import numpy

import _cppad
from _cppad import adouble
from _cppad import addouble



def independent(x):
	"""
	Mark a numpy.array of AD_doubles as independent variables.
	Example:
	import numpy
	import cppad
	x = numpy.array([ad(2.),ad(3.)])
	independent(x)
	"""
	if not isinstance(x,numpy.ndarray):
		raise NotImplementedError('Input has to be of type numpy.array!')
	
	if isinstance(x[0],adouble):
		return _cppad.Independent(x,1)
	elif isinstance(x[0], addouble):
		return _cppad.Independent(x,2)
	else:
		raise NotImplementedError('Only multilevel taping up to 2 is currently implemented!')
	
def ad(x):
	if numpy.isscalar(x):
		return adouble(x)
	elif isinstance(x,adouble):
		return  addouble(x)
	else:
		raise NotImplementedError('Only multilevel taping up to 2 is currently implemented!')



class adfun_double(_cppad.ADFun_double):
	"""
	Create a function object.
	"""
	pass


class adfun_ad_double(_cppad.ADFun_AD_double):
	"""
	Create a function object.
	"""
	pass


def adfun(x,y):
	"""
	Creates a function instance from
	x is a numpy.array of the independent variables
	y is a numpy.array of the   dependent variables

	"""
	if isinstance(x[0], adouble):
		return adfun_double(x,y)
	
	elif isinstance(x[0], addouble):
		return adfun_ad_double(x,y)

	else:
		raise NotImplementedError('Only multilevel taping up to 2 is currently implemented!')		
