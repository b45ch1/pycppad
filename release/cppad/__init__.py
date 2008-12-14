"""\
This is PyCppAD, a Python module to differentiate complex algorithms written in Python.
PyCppAD is a wrapper to CppAD (C++).
"""

import numpy

import _cppad
from _cppad import AD_double



def Independent(x):
	"""
	Mark a vector of AD_doubles as independent variables.
	"""
	if not isinstance(x,numpy.ndarray):
		raise NotImplementedError('Input has to be of type numpy.array')
	return _cppad.Independent(x)

class ADFun_double(_cppad.ADFun_double):
	"""
	Create a function object.
	"""
	pass


