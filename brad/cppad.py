# $begin ad$$ $newlinech #$$
#
# $section Create an Object With One Higher Level of AD$$
#
# $index ad$$
# $index AD, increase level$$
# $index level, increase AD$$
#
# $head Syntax$$
# $icode%a_x% = %ad(%x%)%$$
#
# $head Purpose$$
# Creates an AD object $icode a_x$$ that records floating point operations.
# An $cref/adfun/$$ object can later use this recording to evaluate 
# function values and derivatives. These later evaluations are done
# using the same type as $icode x$$ 
# (except when $icode x$$ is an instance of $code int$$,
# the later evaluations are done using $code float$$ operations).
#
# $head x$$ 
# The argument $icode x$$ must be an instance of an $code int$$ (AD level 0),
# or an instance of $code float$$ (AD level 0),
# or an $code a_float$$ (AD level 1).
#
# $head a_x$$ 
# If $icode x$$ is an instance of $code int$$ or $code float$$,
# $codei a_x$$ is an $code a_float$$ (AD level 1).
# If $icode x$$ is an $code a_float$$,
# $icode a_x$$ is an $code a2float$$ (AD level 2).
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
# $section Create an Object With One Lower Level of AD$$
#
# $head Syntax$$
# $icode%x% = value(%a_x%)%$$
#
# $head Purpose$$
# Returns an object with one lower level of AD recording.
#
# $head a_x$$ 
# The argument $icode a_x$$ must be an $code a_float$$ (AD level 1),
# or an $code a2float$$ (AD level 2).
#
#
# $head x$$ 
# If $icode a_x$$ is an $code a_float$$,
# $icode x$$ is a $code float$$ (AD level 0).
# If $icode a_x$$ is an $code a2float$$,
# $icode x$$ is an $code a_float$$ (AD level 1).
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
# $index independent, variables$$
# $index variables, independent$$
# $index recording, start$$
# $index start, recording$$
#
# $head Purpose$$
# Creates an independent variable vector and starts recording operations
# involving objects that are instances of $icode%type(%a_x%[0])%$$.
# You must create an $cref/adfun/$$ object and stop the recording
#
# $head x$$ 
# The argument $icode x$$ must be a $code numpy.array$$ with one dimension
# (i.e., a vector).
# All the elements of $icode x$$ must all be of the same type and
# instances of either $code int$$, $code float$$ or $code  a_float$$.
#
# $head a_x$$ 
# The return value $icode a_x$$ is a $code numpy.array$$ 
# with the same shape as $icode x$$. 
# If the elements of $icode x$$ are instances of $code int$$ or $code float$$
# the elements of $icode a_x$$ are instances of # $code a_float$$.
# If the elements of $icode x$$ are instances of $code a_float$$
# the elements of $icode a_x$$ are instances of $code a2float$$.
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
# $begin adfun$$ $newlinech #$$
# $spell
#	numpy
#	len
#	adfun
# $$
#
# $section Create a Function Object With One Lower Level of AD$$
#
# $index dependent, variables$$
# $index variables, dependent$$
# $index recording, stop$$
# $index stop, recording$$
#
# $head Syntax$$
# $icode%f% = %adfun%(%a_x%, %a_y%)%$$
#
# $head Purpose$$
# The function object $icode f$$ will store the $codei type( a_x[0] )$$
# operation sequence that mapped the independent variable vector 
# $icode a_x$$ to the dependent variable vector $icode a_y$$.
#
# $head a_x$$
# The argument $icode a_x$$ is the $code numpy.array$$ 
# returned by the previous call to $cref/independent/$$.
# Neither the size of $icode a_x$$, or the value it its elements,
# may change between calling
# $codei%
#	%a_x% = independent(%x%)
# %$$
# and
# $icode%
#	%f% = adfun(%a_x%, %a_y%)
# %$$
# The length of the vector $icode a_x$$ determines the domain size
# $latex n$$ for the function $latex y = F(x)$$ below.
#
# $head a_y$$
# The argument $icode a_y$$ specifies the dependent variables.
# It must be a $code numpy.array$$ with one dimension
# (i.e., a vector) and with the same type of elements as $icode a_x$$.
# The object $icode f$$ stores the $codei type( a_x[0] )$$ operations 
# that mapped the vector $icode a_x$$ to the vector $icode a_y$$.
# The length of the vector $icode a_y$$ determines the range size
# $latex m$$ for the function $latex y = F(x)$$ below.
#
# $head f$$
# The return value $icode f$$ can be used to evaluate the function
# $latex \[
#	F : \B{R}^n \rightarrow \B{R}^m
# \] $$
# and its derivatives, where $latex y = F(x)$$ corresponds to the 
# operation sequence mentioned above.
#
# $children%
#	example/adfun.py
# %$$
# $head Example$$
# The file $cref/adfun.py/$$ 
# contains an example and test of this function.
#
# $end
# ---------------------------------------------------------------------------
# $begin std_math$$ $newlinech #$$
# $spell
#	numpy
#	arccos
#	arcsin
#	arctan
#	cos
#	exp
#	tanh
#	sqrt
# $$
#
# $section Standard Math Unary Functions$$
#
# $index arccos$$
# $index arcsin$$
# $index arctan$$
# $index cos$$
# $index cosh$$
# $index exp$$
# $index log$$
# $index log10$$
# $index sin$$
# $index sinh$$
# $index sqrt$$
# $index tan$$ 
# $index tanh$$
#
# $head Syntax$$
# $icode%y% = %fun%(%x%)%$$
#
# $head Purpose$$
# Evaluate the standard math function $icode fun$$ where $icode fun$$
# has one argument.
#
# $head x$$
# The argument $icode x$$ can be an instance of $code float$$,
# an $code a_float$$, an $code a2float$$, or a $code numpy.array$$
# of such objects.
#
# $head y$$
# If $icode x$$ is an instance of $code float$$,
# $icode y$$ will also be an instance of $icode float$$.
# Otherwise $icode y$$ will have the same type as $icode x$$.
# $pre
#
# $$
# In the case where $icode x$$ is an array, $icode y$$ will 
# the same shape as $icode x$$ and the elements of $icode y$$
# will have the  same type as the elements of $icode x$$.
#
# $head fun$$
# The function $icode fun$$ can be any of the following:
# $code arccos$$,
# $code arcsin$$,
# $code arctan$$,
# $code cos$$,
# $code cosh$$,
# $code exp$$,
# $code log$$,
# $code log10$$,
# $code sin$$,
# $code sinh$$,
# $code sqrt$$,
# $code tan$$, or
# $code tanh$$.
#
# $children%
#	example/std_math.py
# %$$
# $head Example$$
# The file $cref/std_math.py/$$ 
# contains an example and test of these functions.
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
 
def independent(x):
  """
  a_x = independent(x): create independent variable vector a_x, equal to x,
  and start recording operations that use the class corresponding to ad( x[0] ).
  """
  if not isinstance(x, numpy.ndarray):
    raise NotImplementedError('independent(x): x is not of type numpy.array')
  x0 = x[0]
  if isinstance(x0, int) or isinstance(x0, float):
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

