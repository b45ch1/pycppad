"""\
pycppad: a Python algorihtmic differentiation module that uses the C++ package
CppAD to evaluate derivatives of arbitrary order.
"""

import numpy
import pycppad
from numpy import array
from pycppad import a_double
from pycppad import a2double

def ad(x):
  """
  ad(x): returns an object with one higher level of automatic differentiation.
  If x is an int or double (AD level 0), ad(x) is an a_double (AD level 1).
  If x is an a_double (AD level 1), ad(x) is an a2double (AD level 2).
  Higher AD levels for the argument x are not yet supported.
  """
  if numpy.isscalar(x):
    return a_double(x)
  elif isinstance(x, a_double):
    return a2double(x)
  else:
    raise NotImplementedError(
      'ad(x): only implemented where x int, double, or a_double'
    )

def independent(x):
  """
  a_x = independent(x): create independent variable vector a_x, equal to x,
  and start recording operations that use the class corresponding to ad( x[0] ).
  """
  if not isinstance(x, numpy.ndarray):
    raise NotImplementedError('independent(x): x is not of type numpy.array')
  if isinstance(x[0], float):
    return pycppad.independent(x, 1)     # level = 1
  elif isinstance(x[0], a_double):
    return pycppad.independent(x, 2)     # level = 2
  else:
    print "type(x[j]) = ", type(x[0])
    raise NotImplementedError(
      'independent(x): only implemented where x[j] is double or a_double'
    )
class adfun_double(pycppad.adfun_double):
  """
  Create a function object that evaluates using double.
  """
  pass

class adfun_a_double(pycppad.adfun_a_double):
  """
  Create a function object that evaluates using a_double.
  """
  pass

def adfun(x,y):
  """
  f = adfun(x,y): Stop recording and place it in the function object f.
  x: a numpy one dimnesional array containing the independent variable vector.
  y: a vector with same type as x and containing the dependent variable vector.
  """
  if isinstance(x[0], a_double):
    return adfun_double(x, y)
  elif isinstance(x[0], a2double):
    return adfun_a_double(x, y)
  else:
    raise NotImplementedError(
      'adfun(x,y): only implemented where x[j] is a_double or a2double'
    )
