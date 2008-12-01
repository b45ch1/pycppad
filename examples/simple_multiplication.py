#!/usr/bin/env python

import sys
sys.path = ['./release'] + sys.path

from cppad import *
import numpy


# simple multiplication
ax = AD_double(1.5)
ay = AD_double(1.7)
az = ax*ay
print ax
print ay
print az

# numpy arrays of AD_double
ax = numpy.array([AD_double(2 + 1.5*i) for i in range(2)],dtype=object)
Independent(ax)
print ax
ay = numpy.array([ax[0] * ax[1]])
print ay
#x = MyVec(23.)

#test_my_vec(x)