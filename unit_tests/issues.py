import sys
sys.path = ['.'] + sys.path #adding current working directory to the $PYTHONPATH

from cppad import *
import numpy
from numpy import sin


def test_normal_standard_math_syntax_not_supported():
	a_x = ad(1.)
	a_y = a_x.sin()    # supported
	a_z = sin(a_x)     # generates an error 
