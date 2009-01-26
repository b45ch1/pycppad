# $begin ad_numeric.py$$ $newlinech #$$
#
# $section Binary Numeric Operators With an AD Result: Example and Test$$
#
# $index +, example$$
# $index -, example$$
# $index *, example$$
# $index /, example$$
# $index **, example$$
#
# $index example, +$$
# $index example, -$$
# $index example, *$$
# $index example, /$$
# $index example, **$$
#
# $index numeric, binary operator example$$
# $index binary, numeric operator example$$
# $index operator, numeric binary example$$
# $index example, numeric binary operator$$
#
# $code
# $verbatim%example/ad_numeric.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pyad import *
# Example using a_float -----------------------------------------------------
def test_ad_numeric() :
  x    = 2.
  y    = 3.
  a_x  = ad(x)
  a_y  = ad(y)
  #
  assert a_x + a_y == x + y
  assert a_x + y   == x + y
  assert x   + a_y == x + y
  #
  assert a_x - a_y == x - y
  assert a_x - y   == x - y
  assert x   - a_y == x - y
  #
  assert a_x * a_y == x * y
  assert a_x * y   == x * y
  assert x   * a_y == x * y
  #
  assert a_x / a_y == x / y
  assert a_x / y   == x / y
  assert x   / a_y == x / y
  #
  assert a_x ** a_y == x ** y
  assert a_x ** y   == x ** y
  assert x   ** a_y == x ** y
  #
# Example using a2float -----------------------------------------------------
def test_ad_numeric_a2() :
  x    = 2.
  y    = 3.
  a2x  = ad(ad(x))
  a2y  = ad(ad(y))
  #
  assert a2x + a2y == x + y
  assert a2x + y   == x + y
  assert x   + a2y == x + y
  #
  assert a2x - a2y == x - y
  assert a2x - y   == x - y
  assert x   - a2y == x - y
  #
  assert a2x * a2y == x * y
  assert a2x * y   == x * y
  assert x   * a2y == x * y
  #
  assert a2x / a2y == x / y
  assert a2x / y   == x / y
  assert x   / a2y == x / y
  #
  assert a2x ** a2y == x ** y
  assert a2x ** y   == x ** y
  assert x   ** a2y == x ** y
  #
# END CODE
