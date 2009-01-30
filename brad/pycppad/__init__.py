# $begin ad$$ $newlinech #$$
# $spell
#	numpy
# $$
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
# The argument $icode x$$ can be an instance of an $code int$$ (AD level 0),
# or an instance of $code float$$ (AD level 0),
# or an $code a_float$$ (AD level 1).
# The argument $icode x$$ may also be a $code numpy.array$$ with one of the
# element types listed in the previous sentence.
#
# $head a_x$$ 
# If $icode x$$ is an instance of $code int$$ or $code float$$,
# $codei a_x$$ is an $code a_float$$ (AD level 1).
# If $icode x$$ is an $code a_float$$,
# $icode a_x$$ is an $code a2float$$ (AD level 2).
# If $icode x$$ is an $code numpy.array$$,
# $icode a_x$$ is also an $code numpy.array$$ with the 
# same shape as $icode x$$.
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
# $spell
#	numpy
# $$
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
# The argument $icode a_x$$ may also be a $code numpy.array$$ with one of the
# element types listed in the previous sentence.
#
# $head x$$ 
# If $icode a_x$$ is an $code a_float$$,
# $icode x$$ is a $code float$$ (AD level 0).
# If $icode a_x$$ is an $code a2float$$,
# $icode x$$ is an $code a_float$$ (AD level 1).
# If $icode a_x$$ is an $code numpy.array$$,
# $icode x$$ is also an $code numpy.array$$ with the 
# same shape as $icode a_x$$.
# 
# $children%
#	example/value.py
# %$$
# $head Example$$
# The file $cref/value.py/$$ contains an example and test of this function.
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
A Python algorihtmic differentiation module that uses the C++ package CppAD
to evaluate function values and derivatives of arbitrary order.
"""

import numpy
import pycppad.cppad_
from pycppad.cppad_ import a_float
from pycppad.cppad_ import a2float

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
  elif isinstance(x, numpy.ndarray) :
    s      = x.shape
    length = numpy.prod(s)
    a_x    = numpy.array( list(ad(xi) for xi in x.flat) )
    return a_x.reshape(s)
  else :
    raise NotImplementedError(
      'ad(x): only implemented where x an int, float, a_float or '
      'an array of such values.'
    )

def value(a_x) :
  """
  value(a_x): returns object with one lower level of automatic differentation.
  If a_x is an a_float, value(a_x) is a float (AD level 0). 
  If a_x is an a2float, value(a_x) is an a_float (AD level 1). 
  """
  if isinstance(a_x, a_float) :
    return cppad_.float_(a_x);
  elif isinstance(a_x, a2float) :
    return cppad_.a_float_(a_x);
  elif isinstance(a_x, numpy.ndarray) :
    s      = a_x.shape
    length = numpy.prod(s)
    x      = numpy.array( list(value(a_xi) for a_xi in a_x.flat) )
    return x.reshape(s)
  else :
    msg = 'type(a_x) = ' + str( type(a_x) ) + '\n'
    msg += 'value(a_x): only implemented where a_x is an a_float, a2float,\n'
    msg += 'or an array of a_float or a2float'
    raise NotImplementedError(msg)
 
def independent(x) :
  """
  a_x = independent(x): create independent variable vector a_x, equal to x,
  and start recording operations that use the class corresponding to ad( x[0] ).
  """
  if not isinstance(x, numpy.ndarray) :
    raise NotImplementedError('independent(x): x is not of type numpy.array')
  x0 = x[0]
  if isinstance(x0, int) or isinstance(x0, float) :
    return cppad_.independent(x, 1)     # level = 1
  elif isinstance(x0, a_float) :
    return cppad_.independent(x, 2)     # level = 2
  else:
    print "type(x[j]) = ", type(x0)
    raise NotImplementedError(
      'independent(x): only implemented where x[j] is float or a_float'
    )

from adfun import *
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

