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
# involving objects that are instances of $codei%type(%a_x%[0])%$$.
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
# $section Create an AD Function Object$$
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
# The function object $icode f$$ will store the $codei%type( %a_x%[0] )%$$
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
# $codei%
#	%f% = adfun(%a_x%, %a_y%)
# %$$
# The length of the vector $icode a_x$$ determines the domain size
# $latex n$$ for the function $latex y = F(x)$$ below.
#
# $head a_y$$
# The argument $icode a_y$$ specifies the dependent variables.
# It must be a $code numpy.array$$ with one dimension
# (i.e., a vector) and with the same type of elements as $icode a_x$$.
# The object $icode f$$ stores the $codei%type( %a_x%[0] )%$$ operations 
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
# $subhead m$$
# The range size $latex m$$ is equal to the length of the vector $icode a_y$$.
#
# $subhead n$$
# The domain size $latex n$$ is equal to the length of the vector $icode a_x$$.
#
# $subhead level$$
# The $cref/ad/$$ level for the object $icode f$$ is one less than
# the AD level for the arguments $icode a_x$$ and $icode a_y$$;
# i.e., if $codei%type( %a_x%[0] )%$$ is $code a_float$$ ($code a2float$$)
# the corresponding AD level for $icode f$$ is zero (one).
#
# $children%
#	example/adfun.py
# %$$
# $head Example$$
# The file $cref/adfun.py/$$ 
# contains an example and test of this function.
#
# $end

import cppad_
import numpy

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
  elif isinstance(x0, cppad_.a_float) :
    return cppad_.independent(x, 2)     # level = 2
  else:
    print "type(x[j]) = ", type(x0)
    raise NotImplementedError(
      'independent(x): only implemented where x[j] is float or a_float'
    )

class adfun_float(cppad_.adfun_float) :
  """
  Create a level zero function object (evaluates using floats).
  """
  # Kludge: The following reshaping should be done in adfun.cpp
  def jacobian(self, x) :
    return self.jacobian_(x).reshape(self.range(), self.domain())
  def hessian(self, x, w) :
    return self.hessian_(x, w).reshape(self.domain(), self.domain())
  pass

class adfun_a_float(cppad_.adfun_a_float) :
  """
  Create a level one function object (evaluates using a_float).
  """
  # Kludge: The following reshaping should be done in adfun.cpp
  def jacobian(self, x) :
    return self.jacobian_(x).reshape(self.range(), self.domain())
  def hessian(self, x, w) :
    return self.hessian_(x, w).reshape(self.domain(), self.domain())
  pass

def adfun(x,y) :
  """
  f = adfun(x,y): Stop recording and place it in the function object f.
  x: a numpy one dimnesional array containing the independent variable vector.
  y: a vector with same type as x and containing the dependent variable vector.
  """
  if not isinstance(x, numpy.ndarray) or not isinstance(x, numpy.ndarray) :
    raise NotImplementedError('adfun(x, y): x or y is not of type numpy.array')
  x0 = x[0]
  y0 = y[0]
  if isinstance(x0, cppad_.a_float) :
    if isinstance(y0, cppad_.a_float) :
      return adfun_float(x, y)
    else :
      raise NotImplementedError(
        'adfun(x, y): x[j] and y[j] have different elements types')
  elif isinstance(x0, cppad_.a2float) :
    if isinstance(y0, cppad_.a2float) :
      return adfun_a_float(x, y)
    else :
      raise NotImplementedError(
        'adfun(x, y): x[j] and y[j] have different elements types')
  else :
      raise NotImplementedError(
        'adfun(x, y): elements of x and y are not a_float or a2dobule')

