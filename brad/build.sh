#! /bin/bash
# 
python_version="2.5"
python_config_dir="/usr/include/python$python_version"
numpy_dir="/usr/lib/python2.5/site-packages/numpy/core/include"
if [ "$USER" == "bradbell" ]
then
	cppad_dir="$HOME/CppAD/trunk"
else if [ -e "$HOME/workspace/pycppad/cppad-20081128/cppad/cppad.hpp" ]
then
	cppad_dir="../cppad-20081128"
else if [ -e "/u/walter/workspace/PyCPPAD/cppad-20081128/cppad/cppad.hpp" ]
then
	echo "this is wronski"
	cppad_dir="/u/walter/workspace/PyCPPAD/cppad-20081128"
else
	echo "Cannot find cppad/cppad.hpp"
	exit 1
fi
fi
fi
# -------------------------------------------------------------------
if [ ! -e "$python_config_dir/pyconfig.h" ]
then
echo "Must change python_config_dir or python_version in pycppad.sh"
	exit 1
fi
python --version >& pycppad.tmp
py_version=`cat pycppad.tmp`
if ! grep "Python $python_version" pycppad.tmp > /dev/null
then
	echo "Must change python_version in pycppad.sh"
	exit 1
fi
# -------------------------------------------------------------------
echo "# Compile pycppad.cpp --------------------------------------------------" 
#
object_list=""
list="
	vector
	adfun
	pycppad
"
cmd="g++ -fpic -g -c -Wall -I $python_config_dir -I $numpy_dir -I $cppad_dir"
for name in $list
do
	echo "$cmd $name.cpp"
	if ! $cmd $name.cpp
	then
		echo "command failed"
		exit 1
	fi
	object_list="$object_list $name.o"
done
# -------------------------------------------------------------------
echo "# Create pycppad.so dynamic link library ------------------------------"
# needed to link boost python
library_flags="-lboost_python"
cmd="g++ -shared -Wl,-soname,libpycppad.so.1 $library_flags"
cmd="$cmd -o libpycppad.so.1.0 $object_list -lc"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
if [ -e pycppad.so ]
then
	cmd="rm pycppad.so"
	echo $cmd
	$cmd
fi
cmd="ln libpycppad.so.1.0 pycppad.so"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
echo "----------------------------------------------------------------------"
py.test forward_1.py
list="
	forward_2
	compare_op
"
ok="true"
for name in $list
do
	cmd="python $name.py"
	if ! $cmd
	then
		ok="false"
	fi
done
if [ "ok" = "false" ]
then
	echo "At least one test failed."
	exit 1
fi
echo "All the tests passed."
exit 0
