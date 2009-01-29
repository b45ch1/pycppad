# $begin two_levels.py$$ $newlinech #$$
#
# $section Using Two Levels of AD: Example and Test$$ 
#
# $index a2float$$
# $index two, AD levels$$
# $index levels, two AD$$
# $index AD, two levels$$
#
# $head Purpose$$
# This example is intended to demonstrate how 
# $code a_float$$ and $code a2float$$
# can be used together to compute derivatives of functions that are
# defined in terms of derivatives of other functions.
#
# $head F(u)$$
# For this example, the function $latex F : \B{R}^2 \rightarrow  \B{R}$$
# is defined by
# $latex \[
# 	F(u) = u_0^2 + u_1^2
# \] $$
# It follows that
# $latex \[
# \begin{array}{rcl}
#	\partial_{u(0)} F(u) = 2 * u_0 \\
#	\partial_{u(1)} F(u) = 2 * u_1
# \end{array}
# \] $$
#
# $head G(x)$$
# For this example, the function $latex G : \B{R}^2 \rightarrow \B{R}$$ is 
# defined by
# $latex \[
# G(x) = x_1 * \partial_{u(0)} F(x_0 , 1) + x_0 * \partial_{u(1)} F(x_0, 1)
# \] $$
# where $latex \partial{u(j)} F(a, b)$$ denotes the partial of $latex F$$
# with respect to $latex u_j$$ and evaluated at $latex u = (a, b)$$.
# It follows that
# $latex \[
# \begin{array}{rcl}
#	G (x)                 & = & 2 * x_1 * x_0  + 2 * x_0 \\
#	\partial_{x(0)} G (x) & = & 2 * x_1 + 2 \\
#	\partial_{x(1)} G (x) & = & 2 * x_0
# \end{array}
# \] $$
#                         
# $end
from pycppad import *
def pycppad_test_two_levels():
  # start recording a_float operations
  x   = numpy.array( [ 2. , 3. ] )
  a_x = independent(x)

  # start recording a2float operations
  a_u = numpy.array( [a_x[0]  , ad(1) ] )
  a2u = independent(a_u)

  # stop a2float recording and store operations if f
  a2v = numpy.array( [ a2u[0] * a2u[0] + a2u[1] * a2u[1] ] )
  a_f = adfun(a2u, a2v)              # F(u0, u1) = u0 * u0 + u1 * u1

  # evaluate the gradient of F
  a_J = a_f.jacobian(a_u)

  # stop a_float recording and store operations in g
  a_y = numpy.array( [ a_x[1] * a_J[0,0] + a_x[0] * a_J[0,1] ] )
  g   = adfun(a_x, a_y)  # G(x0, x1) = x1 * F_u0(x0, 1) + x0 * F_u1(x0, 1)
  
  # evaluate the gradient of G
  J   = g.jacobian(x)

  assert J[0,0] == 2. * x[1] + 2
  assert J[0,1] == 2. * x[0]
