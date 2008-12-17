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
	

class ad:
	def __init__(self,x):
		if type(x) == float:
			self.level = 1
			self.x  = adouble(x)

		elif isinstance(x,ad):
			if x.level == 1:
				self.x = addouble(x.x)
				self.level = x.level + 1

		else:
			raise ValueError("This type is not recognized!")

	def __str__(self):
		return str(self.x.value)

	# CONDITIONALS
	def __lt__(self,rhs):
		return self.x < rhs.x

	def __le__(self,rhs):
		return self.x <= rhs.x

	def __eq__(self,rhs):
		return self.x == rhs.x

	def __ne__(self,rhs):
		return self.x != rhs.x

	def __ge__(self,rhs):
		return self.x >= rhs.x

	def __gt__(self,rhs):
		return self.x > rhs.x

	# ELEMENTARY OPERATIONS
	def __iadd__(self,rhs):
		self.x += rhs.x
		return self

	def __isub__(self,rhs):
		self.x -= rhs.x
		return self


	def __imul__(self,rhs):
		self.x *= rhs.x
		return self

	def __idiv__(self,rhs):
		self.x /= rhs.x
		return self
	
	def __add__(self,rhs):
		return ad(self.x + rhs.x)

	def __sub__(self,rhs):
		return ad(self.x - rhs.x)

	def __mul__(self,rhs):
		return ad(self.x * rhs.x)

	def __div__(self,rhs):
		return (self.x / rhs.x)

	def __radd__(self,lhs):
		return self + lhs

	def __rsub__(self,lhs):
		return -self + lhs
	
	def __rmul__(self,lhs):
		return self * lhs

	def __rdiv__(self, lhs):
		return lhs/self

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
