from cppad import *
ok = True
one  = ad_double(1)
two  = ad_double(2)
ok = ok and (one < two * 1.)
ok = ok and (two > 1. * one)
ok = ok and (one <= two * two)
ok = ok and (two >= one)
ok = ok and (one != 1. * two)
ok = ok and (one == 1. * one)
if( ok ) :
	print 'OK:    example_3 binary operator test passed.'
else :
	print 'Error: example_3 binary operator test failed.'

	
