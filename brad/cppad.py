"""\
pycppad: a Python algorihtmic differentiation module that uses the C++ package
CppAD to evaluate derivatives of arbitrary order.
"""

import numpy
import python_cppad
from numpy import array
from python_cppad import a_double
from python_cppad import a2double

def independent(x):
  """
  independent(x): mark x as the independent variable vector and start recording
  operations that use the class corresponding to the elements of x.
  """
  if not isinstance(x, numpy.ndarray):
    raise NotImplementedError('independent(x): x is not of type numpy.array')
  if isinstance(x[0], a_double):
    python_cppad.independent(x, 1)     # level = 1
  elif isinstance(x[0], a2double):
    python_cppad.independent(x, 2)     # level = 2
  else:
    raise NotImplementedError(
      'independent(x): only implemented where x[j] is a_double or a2double'
    )
class adfun_double(python_cppad.adfun_double):
  """
  Create a function object that evaluates using double.
  """
  pass

class adfun_a_double(python_cppad.adfun_a_double):
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
