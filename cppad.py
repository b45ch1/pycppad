"""\
This is PyCppAD, a Python module to differentiate complex algorithms written in Python.
PyCppAD is a wrapper to CppAD (C++).
"""

import numpy

import _cppad
from _cppad import AD_double as ad_double
from _cppad import AD_AD_double as ad_ad_double
from _cppad import ad

def independent(x):
	"""
	Mark a vector of AD_doubles as independent variables.
	"""
	print 'called Independent'
	if not isinstance(x,numpy.ndarray):
		raise NotImplementedError('Input has to be of type numpy.array!')
	
	if x[0].__class__ == _cppad.AD_double:
		print 'level = 1'
		return _cppad.Independent(x,1)
	elif x[0].__class__ == _cppad.AD_AD_double:
		print 'level = 2'
		return _cppad.Independent(x,2)
	elif isinstance(x[0], ad):
		return _cppad.Independent(x,x[0].level)
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
