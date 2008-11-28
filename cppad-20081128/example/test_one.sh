# -----------------------------------------------------------------------------
# CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-08 Bradley M. Bell
#
# CppAD is distributed under multiple licenses. This distribution is under
# the terms of the 
#                     Common Public License Version 1.0.
#
# A copy of this license is included in the COPYING file of this distribution.
# Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
# -----------------------------------------------------------------------------
# Run one of the tests
if [ "$1" = "" ]
then
	echo "usage: test_one.sh file [extra]"
	echo "file is the *.cpp file name with extension"
	echo "and extra is extra options for g++ command"
	exit 1
fi
# determine the function name
fun=`grep "^bool *[a-zA-Z_]*( *void *)" $1 | tail -1 | \
	sed -e "s/^bool *\([a-zA-Z_]*\) *( *void *)/\1/"`
#
if [ -e test_one.exe ]
then
	rm test_one.exe
fi
if [ -e test_one.cpp ]
then
	rm test_one.cpp
fi
sed < example.cpp > test_one.cpp \
-e '/ok *\&= *Run( /d' \
-e "s/.*This line is used by test_one.sh.*/	ok \&= Run( $fun, \"$fun\");/"  
#
cmd="g++ test_one.cpp $*
	-o test_one.exe
	-g -Wall -ansi -pedantic-errors 
	-std=c++98 -DCPPAD_ADOLC_EXAMPLES
	-I.. -I/usr/include/boost-1_33_1 
	-I$HOME/include
"
if [ -e /include/adolc/adouble.h ]
then
	cmd="$cmd -I/include -L/lib -ladolc"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lib"
fi
if [ -e /lib/libipopt.a ]
then
	cmd="$cmd -I/include"
	cmd="$cmd -L/lib -lipopt"
	cmd="$cmd  -L/usr/lib/gcc/i586-suse-linux/4.2.1 -L/usr/lib/gcc/i586-suse-linux/4.2.1/../../../../i586-suse-linux/lib -L/usr/lib/gcc/i586-suse-linux/4.2.1/../../.. -lgfortranbegin -lgfortran -lm -lgcc_s -lpthread -ldl"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lib"
fi
echo $cmd
$cmd
#
echo "./test_one.exe"
if ! ./test_one.exe
then
	exit 1
fi
exit 0

