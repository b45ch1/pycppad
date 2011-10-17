# $begin condexp.py$$ $newlinech #$$
# $spell
#	condexp
# $$
#
# $section condexp: Example and Test$$
#
# $index conditional, expression$$
# $index condexp, example$$
# $index example, condexp$$
#
# $code
# $verbatim%example/condexp.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
# Example using a_float ----------------------------------------------------
from pycppad import *
def pycppad_test_condexp() :
  x          = numpy.array( [1. , 1., 3., 4. ] )
  a_x        = independent(x)
  a_left     = a_x[0];
  a_right    = a_x[1];
  a_if_true  = a_x[2];
  a_if_false = a_x[3]; 
  a_y_lt     = condexp_lt(a_left, a_right, a_if_true, a_if_false);
  a_y_le     = condexp_le(a_left, a_right, a_if_true, a_if_false);
  a_y_eq     = condexp_eq(a_left, a_right, a_if_true, a_if_false);
  a_y_ge     = condexp_ge(a_left, a_right, a_if_true, a_if_false);
  a_y_gt     = condexp_gt(a_left, a_right, a_if_true, a_if_false);
  a_y        = numpy.array( [ a_y_lt, a_y_le, a_y_eq, a_y_ge, a_y_gt ] );
  f          = adfun(a_x, a_y)
  y          = f.forward(0, x)
  assert ( y[0] == 4. )  # 1 <  1 is false so result is 4
  assert ( y[1] == 3. )  # 1 <= 1 is true  so result is 3
  assert ( y[2] == 3. )  # 1 == 1 is true  so result is 3
  assert ( y[3] == 3. )  # 1 >= 1 is true  so result is 3
  assert ( y[4] == 4. )  # 1 >  2 is false so result is 4
  x          = numpy.array( [4., 3., 2., 1.] )
  y          = f.forward(0, x)
  assert ( y[0] == 1. )  # 4 <  3 is false so result is 1
  assert ( y[1] == 1. )  # 4 <= 3 is false so result is 1
  assert ( y[2] == 1. )  # 4 == 3 is false so result is 1
  assert ( y[3] == 2. )  # 4 >= 3 is true  so result is 2
  assert ( y[4] == 2. )  # 4 >  3 is true  so result is 2
# Example using a2float ----------------------------------------------------
def pycppad_test_condexp_a2() :
  x          = numpy.array( [1. , 1., 3., 4. ] )
  a_x        = ad(x)
  # begin level two recording of conditional expression
  a2x        = independent(a_x)
  a2left     = a2x[0];
  a2right    = a2x[1];
  a2if_true  = a2x[2];
  a2if_false = a2x[3]; 
  a2y_lt     = condexp_lt(a2left, a2right, a2if_true, a2if_false);
  a2y_le     = condexp_le(a2left, a2right, a2if_true, a2if_false);
  a2y_eq     = condexp_eq(a2left, a2right, a2if_true, a2if_false);
  a2y_ge     = condexp_ge(a2left, a2right, a2if_true, a2if_false);
  a2y_gt     = condexp_gt(a2left, a2right, a2if_true, a2if_false);
  a2y        = numpy.array( [ a2y_lt, a2y_le, a2y_eq, a2y_ge, a2y_gt ] );
  a_f        = adfun(a2x, a2y)
  # begin level one recording of conditional expression
  a_x        = independent(x)
  a_y        = a_f.forward(0, a_x)
  f          = adfun(a_x, a_y)
  y          = f.forward(0, x)
  assert ( y[0] == 4. )  # 1 <  1 is false so result is 4
  assert ( y[1] == 3. )  # 1 <= 1 is true  so result is 3
  assert ( y[2] == 3. )  # 1 == 1 is true  so result is 3
  assert ( y[3] == 3. )  # 1 >= 1 is true  so result is 3
  assert ( y[4] == 4. )  # 1 >  2 is false so result is 4
  x          = numpy.array( [4., 3., 2., 1.] )
  y          = f.forward(0, x)
  assert ( y[0] == 1. )  # 4 <  3 is false so result is 1
  assert ( y[1] == 1. )  # 4 <= 3 is false so result is 1
  assert ( y[2] == 1. )  # 4 == 3 is false so result is 1
  assert ( y[3] == 2. )  # 4 >= 3 is true  so result is 2
  assert ( y[4] == 2. )  # 4 >  3 is true  so result is 2
# END CODE
