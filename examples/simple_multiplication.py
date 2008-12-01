#!/usr/bin/env python
import sys
sys.path = ['./release'] + sys.path

from cppad import *
import numpy

# numpy arrays of AD_double
ax = numpy.array([AD_double(2 + 1.5*i) for i in range(2)],dtype=object)
Independent(ax)
ay = numpy.array([ax[0] * ax[1]])
# f = ADFun_double(ax,ay)
#f.forward(...)
#f.reverse(...)


