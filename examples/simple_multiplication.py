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
av = numpy.array([AD_double(1.5*i) for i in range(10)],dtype=object)
Independent(av)

#x = MyVec(23.)

#test_my_vec(x)