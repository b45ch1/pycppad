#!/usr/bin/env python

import sys
sys.path = ['../release'] + sys.path

from cppad import *


ax = AD_double(3.)
ay = AD_double(2.)
az = ax*ay
print ax
print ay
print az