"""\
This is PyCppAD, a Python module to differentiate complex algorithms written in Python.
PyCppAD is a wrapper to CppAD (C++).

Example:
	x = numpy.array([ad(2.), ad(3.)])
	y = numpy.array([x[0]*x[1]])
	independent(x)
	adf = adfun(x,y)

	# univariate Taylor propagation
	f = adf.forward(0, numpy.array([7.,13.]))
	g1 = adf.forward(1, numpy.array([1., 0.]))
	g2 = adf.forward(1, numpy.array([0., 1.]))
	
	# high level functions
	H = adf.hessian(numpy.array([7.,13.]))

"""

import numpy

import _cppad
from _cppad import a_double
from _cppad import a_double as a1double
from _cppad import a2double

def independent(x):
	"""
	Mark an 1D numpy.array of a_doubles or a2doubles as independent variables and start recording.
	"""
	if not isinstance(x,numpy.ndarray):
		raise NotImplementedError('Input has to be of type numpy.array!')
	
	if isinstance(x[0],a_double):
		return _cppad.Independent(x,1)
	elif isinstance(x[0], a2double):
		return _cppad.Independent(x,2)
	else:
		raise NotImplementedError('Only multilevel taping up to 2 is currently implemented!')
	
def ad(x):
	"""
	ad(x): returns an object with one higher level of automatic differentiation.
	If x is an int or double (AD level 0), ad(x) is an a_double (AD level 1).
	If x is an a_double (AD level 1), ad(x) is an a2double (AD level 2).
	Higher AD levels for the argument x are not yet supported.
	"""	
	if numpy.isscalar(x):
		return a_double(x)
	elif isinstance(x,a_double):
		return  a2double(x)
	else:
		raise NotImplementedError('Only multilevel taping up to 2 is currently implemented!')


class adfun_double(_cppad.ADFun_double):
	"""
	Create a function object.
	"""
	def hessian(self, x, w = None):
		if w == None:
			w = numpy.array([1.],dtype=float)

		if not isinstance(x,numpy.ndarray):
			raise NotImplementedError('Input has to be of type numpy.array!')

		if not isinstance(w,numpy.ndarray):
			raise NotImplementedError('Input has to be of type numpy.array!')

		return self.lagrange_hessian(x,w)


class adfun_ad_double(_cppad.ADFun_AD_double):
	"""
	Create a function object.
	"""
	pass


def adfun(x,y):
	"""
	f = adfun(x,y)
	adfun: Stops recording creates a function object f
	f: function object
	x: a 1D numpy.array with elements of type float or a_double, a2double,...
	y: a 1D numpy.array with elements of the same type as x
	"""
	if isinstance(x[0], a_double):
		return adfun_double(x,y)
	
	elif isinstance(x[0], a2double):
		return adfun_ad_double(x,y)

	else:
		raise NotImplementedError('Only multilevel taping up to 2 is currently implemented!')		
