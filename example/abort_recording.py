# $begin abort_recording.py$$ $newlinech #$$
#
# $section abort_recording: Example and Test$$
#
# $index abort_recording, example$$
# $index example, abort_recording$$
#
# $code
# $verbatim%example/abort_recording.py%0%# BEGIN CODE%# END CODE%1%$$
# $$
# $end
# BEGIN CODE
from pycppad import *
# Example using a_float ---------------------------------------------------
def pycppad_test_abort_recording() :
	from numpy import array
	try :
		x    = numpy.array( [ 1., 2., 3. ] )
		a_x  = independent(x)    # start first level recording
		a2_x = independent(a_x)  # start second level recording
		a_y  = array([sum(a_x)]) # record some operations
		if a_y[0] > 2 :
			raise ValueError
	except ValueError :
		# Pretend that we are not sure if there are any active recordings
		# and use this call to terminate any that may exist.
		abort_recording()

	a_x  = independent(x)     # test starting a level 1 recording
	a2_x = independent(a_x)   # test starting a level 2 recording
	a_y  = array([sum(a_x)])  # record some level 1 operations
	f    = adfun(a_x, a_y)    # terminate level 1 recording
	y    = f.forward(0, x)    # evaluate the function at original x value 
	assert( y[0] == 6. )      # check the value
	abort_recording()         # abort the level 2 recording

# END CODE
