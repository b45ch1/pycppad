#! /usr/bin/env python
#
import sys
import os
# need this directory to import example files 
sys.path.append(os.getcwd() + '/example')
# list of files in example directory
example_list = os.listdir('example')
# import each example file
for file_name in example_list :
	if file_name.endswith('.py') :
		module = file_name[:-3]
		exec('from ' + module + ' import *')
#
list_of_globals = globals().keys()
number_ok   = 0
number_fail = 0
for name in list_of_globals :
	if name[:13] == "pycppad_test_" :
		ok = True
		try :
			eval("%s()" % name)
		except AssertionError :
			ok = False
		if ok : 
			print "OK:    %s" % name[13:]
			number_ok = number_ok + 1
		else : 
			print "Error: %s" % name[13:]
			number_fail = number_fail + 1
if number_fail == 0 : 
	print "All %d tests passed" % number_ok
	sys.exit(0)
else :
	print "%d tests failed" % number_fail 
	sys.exit(1)
