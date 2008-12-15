"""\
PyCppAD: a Python algorihtmic differentiation module.
(PyCppAD is a wrapper to the C++ AD package CppAD).
"""

import numpy
import python_cppad
from numpy import array
from python_cppad import ad_double
from python_cppad import ad_ad_double



def independent(x, level):
	"""
	Mark a vector of AD_doubles as independent variables.
	"""
	if not isinstance(x, numpy.ndarray):
		raise NotImplementedError('Input has to be of type numpy.array')
	return python_cppad.independent(x, level)

class adfun_double(python_cppad.adfun_double):
	"""
	Create a function object that evaluates using double.
	"""
	pass

class adfun_ad_double(python_cppad.adfun_ad_double):
	"""
	Create a function object that evaluates using ad_double.
	"""
	pass


