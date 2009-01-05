# $begin ad$$ $newlinech #$$
#
# $section Create an object with one higher level of AD$$
#
# $index ad$$
# $index AD, increase level$$
# $index level, increase AD$$
#
# $head Syntax$$
# $codei%ad(%x%)%$$
#
# $head Purpose$$
# Creates an object with an AD type that records floating point operations.
# An $cref/adfun/$$ object can later use this recording to evaluate 
# function values and derivatives. These later evaluations are done
# using the same type as $icode x$$ 
# (except when $icode x$$ is an $code int$$ the later evaluations are done
# using $code float$$ operations).
#
# $head x$$ 
# If $icode x$$ is an $code int$$ or $code float$$ (an AD level 0 value),
# $codei%ad(%x%)%$$ is an $code a_float$$  (an AD level 1 value).
# If $icode x$$ is an $code a_float$$ (an AD level 1 value),
# $codei%ad(%x%)%$$ is an $code a2float$$  (an AD level 2 value).
#
# 
# $children%
#	example/ad.py
# %$$
# $head Example$$
# The file $cref/ad.py/$$ contains an example and test of this function.
#
# $end
# ---------------------------------------------------------------------------
# $begin value$$ $newlinech #$$
#
# $index value$$
# $index AD, decrease level$$
# $index level, decrease AD$$
#
# $section Create an object with one lower level of AD$$
#
# $head Syntax$$
# $codei%value(%x%)%$$
#
# $head Purpose$$
# Creates an object with one lower level of AD recording.
#
# $head x$$ 
# If $icode x$$ is an $code a_float$$ (an AD level 1 value),
# $codei%value(%x%)%$$ is an $code float$$  (an AD level 0 value).
# If $icode x$$ is an $code a2float$$ (an AD level 2 value),
# $codei%value(%x%)%$$ is an $code a_float$$  (an AD level 1 value).
#
# 
# $children%
#	example/value.py
# %$$
# $head Example$$
# The file $cref/value.py/$$ contains an example and test of this function.
#
# $end
# ---------------------------------------------------------------------------
# $begin array$$ $newlinech #$$
# $spell
#	Numpy
#	tuple
#	ndarray
# $$
#
# $section Create a Numpy N-Dimensional Array object$$
#
# $head Syntax$$
# $codei%array(%x%)%$$
#
# $head Purpose$$
# Creates a 
# $href%http://numpy.scipy.org/%numpy%$$ n-dimensional array object.
#
# $head x$$ 
# The argument $icode x$$ must be either a python list or a tuple.
# If $icode%x%[0]%$$ is an $code int$$ or $code float$$,
# $codei%array(%x%)%$$ is corresponding $code numpy.ndarray$$ object
# with elements that are instances of $code float$$.
# Otherwise, it is the corresponding Numpy array with elements
# that are instances of $codei%type(%x%[0])%$$.
# 
# $children%
#	example/array.py
# %$$
# $head Example$$
# The file $cref/array.py/$$ contains an example and test of this function.
#
# $end
# ---------------------------------------------------------------------------
# $begin independent$$ $newlinech #$$
# $spell
#	Numpy
#	tuple
# $$
#
# $section Create an Independent Variable Vector$$
#
# $head Syntax$$
# $icode%a_x% = independent(%x%)%$$
#
# $head Purpose$$
# Creates an independent variable vector and starts recording operations
# involving objects that are instances of $icode%type(%a_x%[0])%$$.
# You must create an $cref/adfun/$$ object and stop the recording
#
# $head x$$ 
# The argument $icode x$$ must be a Numpy $cref/array/$$.
# The elements of $icode x$$ must all be of the same type and
# instances of either $code float$$ or $code  a_float$$.
# There can only be one dimension to the array; i.e., it must be a vector.
#
# $head a_x$$ 
# The return value $icode a_x$$ is a Numpy $cref/array/$$ with the same shape 
# as $icode x$$. If the elements of $icode x$$ are instances of $code float$$
# ($code a_float$$) the elements of $icode a_x$$ are instances of 
# $code a_float$$ ($code a2float$$).
# The $cref/value/$$ of the elements of $icode a_x$$ 
# are equal to the corresponding elements of $icode x$$.
# 
# $children%
#	example/independent.py
# %$$
# $head Example$$
# The file $cref/independent.py/$$ 
# contains an example and test of this function.
#
# $end
# ---------------------------------------------------------------------------
"""\
pycppad: a Python algorihtmic differentiation module that uses the C++ package
CppAD to evaluate derivatives of arbitrary order.
"""

import numpy

import pycppad
from pycppad import a_float
from pycppad import a2float

def ad(x) :
  """
  ad(x): returns an object with one higher level of automatic differentiation.
  If x is an int, or float (AD level 0), ad(x) is an a_float 
  (AD level 1).  If x is an a_float (AD level 1), ad(x) is an a2float 
  (AD level 2).  Higher AD levels for the argument x are not yet supported.
  """
  if isinstance(x, int) or isinstance(x, float) :
    return a_float(x)
  elif isinstance(x, a_float) :
    return a2float(x)
  else:
    raise NotImplementedError(
      'ad(x): only implemented where x int, float, or a_float'
    )

def value(x) :
  """
  value(x): returns an object with one lower level of automatic differentation.
  If x is an a_float, value(x) is a float (AD level 0). 
  If x is an a2float, value(x) is an a_float (AD level 1). 
  """
  if isinstance(x, a_float) :
    return pycppad.float_(x);
  elif isinstance(x, a2float) :
    return pycppad.a_float_(x);
  else :
    raise NotImplementedError(
      'value(x): only implemented where x a_float or a2float'
    )
 
def array(x) :
  """
  array(x): converts a list or tuple to an array.
  If the elements of x are int or float, the elements of the array are floats.
  Otherwise the elements of the array are the same type as the elements of x.
  """
  x0 = x[0]
  if isinstance(x0, int) or isinstance(x0, float) :
    return numpy.asarray(x, dtype=float)
  return numpy.asarray(x, type(x0))

def independent(x):
  """
  a_x = independent(x): create independent variable vector a_x, equal to x,
  and start recording operations that use the class corresponding to ad( x[0] ).
  """
  if not isinstance(x, numpy.ndarray):
    raise NotImplementedError('independent(x): x is not of type numpy.array')
  x0 = x[0]
  if isinstance(x0, float) :
    return pycppad.independent(x, 1)     # level = 1
  elif isinstance(x0, a_float) :
    return pycppad.independent(x, 2)     # level = 2
  else:
    print "type(x[j]) = ", type(x0)
    raise NotImplementedError(
      'independent(x): only implemented where x[j] is float or a_float'
    )

class adfun_float(pycppad.adfun_float) :
  """
  Create a function object that evaluates using floats.
  """
  def jacobian(self, x) :
    return self.jacobian_(x).reshape(self.range(), self.domain())
  pass

class adfun_a_float(pycppad.adfun_a_float) :
  """
  Create a function object that evaluates using a_float.
  """
  def jacobian(self, x) :
    return self.jacobian_(x).reshape(self.range(), self.domain())
  pass

def adfun(x,y):
  """
  f = adfun(x,y): Stop recording and place it in the function object f.
  x: a numpy one dimnesional array containing the independent variable vector.
  y: a vector with same type as x and containing the dependent variable vector.
  """
  if not isinstance(x, numpy.ndarray) or not isinstance(x, numpy.ndarray) :
    raise NotImplementedError('adfun(x, y): x or y is not of type numpy.array')
  x0 = x[0]
  y0 = y[0]
  if isinstance(x0, a_float) :
    if isinstance(y0, a_float) :
      return adfun_float(x, y)
    else :
      raise NotImplementedError(
        'adfun(x, y): x[j] and y[j] have different elements types')
  elif isinstance(x0, a2float) :
    if isinstance(y0, a2float) :
      return adfun_a_float(x, y)
    else :
      raise NotImplementedError(
        'adfun(x, y): x[j] and y[j] have different elements types')
  else :
      raise NotImplementedError(
        'adfun(x, y): elements of x and y are not a_float or a2dobule')


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

# Kludge: for some reason numpy.sin(x) will work for an array of a_float
# but numpy.abs(x) will not work for an array of a_float.
def abs(x) :
  if isinstance(x, a_float) or isinstance(x, a2float) :
    return x.abs()
  if isinstance(x, numpy.ndarray) :
    n  = len(x)
    x0 = x[0]
    if isinstance(x0, a_float) :
      a_zero = ad(0)
      y = array( list( a_zero for i in range(n) ) )
      for i in range(n) :
         y[i] = x[i].abs()
      return y
    if isinstance(x0, a2float) :
      a2zero = ad(0)
      y = array( list( a2zero for i in range(n) ) )
      for i in range(n) :
         y[i] = x[i].abs()
      return y
  return numpy.abs(x)
