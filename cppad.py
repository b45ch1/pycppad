"""\
This is PyCppAD, a Python module to differentiate complex algorithms written in Python.
PyCppAD is a wrapper to CppAD (C++).
"""

import numpy

import _cppad
from _cppad import AD_double as adouble
from _cppad import AD_AD_double as addouble



def independent(x):
	"""
	Mark a vector of AD_doubles as independent variables.
	"""
	if not isinstance(x,numpy.ndarray):
		raise NotImplementedError('Input has to be of type numpy.array!')
	
	if isinstance(x[0],_cppad.AD_double):
		return _cppad.Independent(x,1)
	elif isinstance(x[0], _cppad.AD_AD_double):
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
	if isinstance(x[0], adouble):
		return adfun_double(x,y)
	
	elif isinstance(x[0], addouble):
		return adfun_ad_double(x,y)

	else:
		raise NotImplementedError('Only multilevel taping up to 2 is currently implemented!')		
