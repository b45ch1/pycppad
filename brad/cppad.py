"""\
pycppad: a Python algorihtmic differentiation module that uses the C++ package
CppAD to evaluate derivatives of arbitrary order.
"""

import numpy
from numpy import double

from numpy import arccos
from numpy import arcsin
from numpy import arctan
from numpy import cos
from numpy import cosh
from numpy import exp
from numpy import log
from numpy import log10
from numpy import sin
from numpy import sinh
from numpy import sqrt
from numpy import tan
from numpy import tanh

import pycppad
from pycppad import a_double
from pycppad import a2double

def ad(x):
  """
  ad(x): returns an object with one higher level of automatic differentiation.
  If x is an int, float, or double (AD level 0), ad(x) is an a_double 
  (AD level 1).  If x is an a_double (AD level 1), ad(x) is an a2double 
  (AD level 2).  Higher AD levels for the argument x are not yet supported.
  """
  type_x = type(x)
  if type_x == int or type_x == float or type_x == double :
    return a_double(x)
  elif type_x == a_double :
    return a2double(x)
  else:
    raise NotImplementedError(
      'ad(x): only implemented where x int, double, or a_double'
    )

def array(x) :
  """
  array(x): converts a list or tuple to an array.
  If the elements of x are int, float, or double, the elements of the 
  array are doubles.
  Otherwise the elements of the array are the same type as the elements of x.
  """
  type_x0 = type(x[0])
  if type_x0 == int or type_x0 == float or type_x0 == double :
    return numpy.asarray(x, dtype=double)
  return numpy.asarray(x, dtype=type_x0)

def independent(x):
  """
  a_x = independent(x): create independent variable vector a_x, equal to x,
  and start recording operations that use the class corresponding to ad( x[0] ).
  """
  if not isinstance(x, numpy.ndarray):
    raise NotImplementedError('independent(x): x is not of type numpy.array')
  type_x0 = type(x[0])
  if type_x0 == double :
    return pycppad.independent(x, 1)     # level = 1
  elif type_x0 == a_double :
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
  if not isinstance(x, numpy.ndarray) or not isinstance(x, numpy.ndarray) :
    raise NotImplementedError('adfun(x, y): x or y is not of type numpy.array')
  type_x0 = type(x[0])
  type_y0 = type(y[0])
  if type_x0 != type_y0 :
    raise NotImplementedError(
      'adfun(x, y): x and y have different elements types')
  if type_x0 ==  a_double :
    return adfun_double(x, y)
  elif type_x0 ==  a2double :
    return adfun_a_double(x, y)
  else:
    raise NotImplementedError(
      'adfun(x,y): only implemented where x[j] is a_double or a2double')
