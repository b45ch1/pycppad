from cppad import *
ok = True
one  = a_double(1)
two  = a_double(2)
ok = ok and (one < two * 1.)
ok = ok and (two > 1. * one)
ok = ok and (one <= two * two)
ok = ok and (two >= one)
ok = ok and (one != 1. * two)
ok = ok and (one == 1. * one)
if( ok ) :
	print 'OK:    compare_op: comparision operators.'
else :
	print 'Error: compare_op: comparision operators.'

	
