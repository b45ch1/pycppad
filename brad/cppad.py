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
  x: is an int, double (level zero AD), or a_double (level one AD).
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
  independent(x): mark x as the independent variable vector and start recording
  operations that use the class corresponding to the elements of x.
  """
  if not isinstance(x, numpy.ndarray):
    raise NotImplementedError('independent(x): x is not of type numpy.array')
  if isinstance(x[0], a_double):
    pycppad.independent(x, 1)     # level = 1
  elif isinstance(x[0], a2double):
    pycppad.independent(x, 2)     # level = 2
  else:
    raise NotImplementedError(
      'independent(x): only implemented where x[j] is a_double or a2double'
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
