# $begin runge_kutta_4$$ $newlinech #$$
# $spell
#	pycppad
#	def
#	numpy
#	yi
#	yf
#	Runge-Kutta
#	dt
#	2nd ed.
# $$
# $index ODE solver, Runge-Kutta$$
# $index Runge-Kutta, ODE solver$$
#
# $section Fourth Order Runge Kutta$$
#
# $head Syntax$$
# $codei% yf = runge_kutta_4(%f%, %ti%, %yi%, %dt%)%$$
#
# $head Purpose$$
# See $cref/Motivation/runge_kutta_4/Motivation/$$ below 
# as well as the purpose described here.
# We are given a function $latex f : \B{R}^n \rightarrow \B{R}^n$$,
# and a point $latex yi \in \B{R}^n$$ such that an unknown function
# $latex y : \B{R} \rightarrow \B{R}^n $$ satisfies the equations
# $latex \[
#	\begin{array}{rcl}
#		y( ti ) & = & yi \\
#		y'(t)    & = & f[t, y(t) ] \\
#	\end{array}
# \] $$
# We use the Fourth order Runge-Kutta formula (see equation 16.1.2 of 
# Numerical Recipes in Fortran, 2nd ed.) wish to approximate the value of 
# $latex \[
#	yf = y( ti + \Delta t )
# \] $$
#
# $head f$$
# If $icode t$$ is a scalar and $icode y$$ is a vector with size $latex n$$,
# $icode%
#	k = f(t, y)
# %$$
# returns a vector of size $latex n$$ that is the value of $latex f(t, y)$$
# at the specified values.
#
# $head ti$$
# is a scalar that specifies the value of $latex ti$$ in the problem above. 
#
# $head yi$$
# is a vector of size $latex n$$ that specifies the value of 
# $latex yi$$ in the problem above. 
#
# $head dt$$
# is a scalar that specifies the value of $latex \Delta t$$ 
# in the problem above. 
#
# $head yf$$
# is a vector of size $latex n$$ that is the approximation for
# $latex y( t + \Delta t )$$.
#
# $head Motivation$$
# This routine makes very few assumptions about the objects used to do these
# calculations. Thus, smart objects can be used for all sorts of purposes;
# for example, computing derivatives of the solution of an ODE.
# The table below lists the assumptions on the objects passed into
# $code runge_kutta_4$$. In this table, $icode s$$ and $icode t$$ are
# scalars, $icode d$$ is a decimal number (i.e., a $code float$$)
# and $icode u$$ and $icode v$$ are vectors with size $latex n$$.
# $table
# $bold operation$$    $cnext $bold result$$  $rnext
# $icode%d% * %s%$$    $cnext scalar                      $rnext 
# $icode%s% + %t%$$    $cnext scalar                      $rnext 
# $icode%s% * %u%$$    $cnext vector with size $latex n$$ $rnext 
# $icode%d% * %u%$$    $cnext vector with size $latex n$$ $rnext 
# $icode%s% * %u%$$    $cnext vector with size $latex n$$ $rnext 
# $icode%u% + %v%$$    $cnext vector with size $latex n$$ $rnext 
# $tend
#
# $head Source Code$$
# $codep
def runge_kutta_4(f, ti, yi, dt) :
	k1 = dt * f(ti         , yi)
	k2 = dt * f(ti + .5*dt , yi + .5*k1) 
	k3 = dt * f(ti + .5*dt , yi + .5*k2) 
	k4 = dt * f(ti + dt    , yi + k3)
	yf = yi + (1./6.) * ( k1 + 2.*k2 + 2.*k3 + k4 )
	return yf 
# $$
#
#
# $children%
#	example/runge_kutta_4_correct.py%
#	example/runge_kutta_4_ad.py%
#	example/runge_kutta_4_cpp.py
# %$$
# $head Example$$
# $list number$$
# The file $cref/runge_kutta_4_correct.py/$$ contains an example and test of
# using $code runge_kutta_4$$ to solve an ODE.
# $lnext
# The file $cref/runge_kutta_4_ad.py/$$ contains an example and test of
# differentiating the numerical solution of an ODE.
# $lnext
# The file $cref/runge_kutta_4_cpp.py/$$ contains an example and test of
# using pycppad $cref/adfun/$$ object to evaluate python functions with
# C++ speed of execution.
# $lend
#
# $end
