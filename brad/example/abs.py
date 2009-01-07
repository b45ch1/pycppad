# $begin abs.py$$ $newlinech #$$
#
# $section abs: Example and Test$$
#
# $index abs, example$$
# $index example, abs$$
#
# $head Theory$$
# Define $latex F(x) = \R{abs}(x)$$. It follows that
# $latex \[
#	F^{(1)} (x) = \left\{ \begin{array}{ll} 
#		1 & \R{if} \; x > 0
#		\\
#		-1 & \R{if} \; x < 0
#	\end{array} \right.
# \] $$
# and the derivative $latex F^{(1)} (0)$$ does not exist.
# On the other hand, the directional derivatives
# $latex \[
#	F^\circ ( x , d ) = \lim_{\lambda \downarrow 0 } 
#		\frac{F(x + \lambda d) - F(x) }{ \lambda }
# \] $$ 
# exists for all $latex x$$ and all $latex d$$. For $latex x \neq 0$$,
# $latex \[
#	F^\circ ( x , d ) = F^{(1)} ( x ) * d
# \] $$
# and $latex F^\circ (0 , 1) =  F^\circ (0, -1) = 1$$.
# 
# $code
# $verbatim%example/abs.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from cppad import *
#
def test_abs() :
  x   = array( [ -1.,  0.,  1.] )
  n   = len(x)
  a_x = independent(x)
  a_y = abs( a_x )
  f   = adfun(a_x, a_y)
  f.forward(0, x)
  dx  = numpy.zeros(n, dtype=float)
  for i in range( n ) :
    dx[i] = 1.
    df    = f.forward(1, dx)
    if x[i] >= 0 :
      assert df[i] == +1.
    else :
      assert df[i] == -1.
    dx[i] = -1.
    df    = f.forward(1, dx)
    if x[i] > 0 :
      assert df[i] == -1.
    else :
      assert df[i] == +1.
    dx[i] = 0.
# END CODE
