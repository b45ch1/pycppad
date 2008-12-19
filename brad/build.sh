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
echo "Must change python_config_dir or python_version in python_cppad.sh"
	exit 1
fi
python --version >& python_cppad.tmp
py_version=`cat python_cppad.tmp`
if ! grep "Python $python_version" python_cppad.tmp > /dev/null
then
	echo "Must change python_version in python_cppad.sh"
	exit 1
fi
# -------------------------------------------------------------------
echo "# Compile python_cppad.cpp -------------------------------------------" 
#
object_list=""
list="
	vector
	adfun
	python_cppad
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
echo "# Create python_cppad.so dynamic link library -----------------------"
# needed to link boost python
library_flags="-lboost_python"
cmd="g++ -shared -Wl,-soname,libpython_cppad.so.1 $library_flags"
cmd="$cmd -o libpython_cppad.so.1.0 $object_list -lc"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
if [ -e cppad/python_cppad.so ]
then
	cmd="rm cppad/python_cppad.so"
	echo $cmd
	$cmd
fi
cmd="ln libpython_cppad.so.1.0 cppad/python_cppad.so"
echo $cmd
if ! $cmd
then
	echo "command failed"
	exit 1
fi
# -------------------------------------------------------------------
echo "-------------------------------------------------------------------"
cmd="python example_1.py"
echo $cmd
if ! $cmd
then
	echo "example_1 failed"
	exit 1
fi
echo "-------------------------------------------------------------------"
cmd="python example_2.py"
echo $cmd
if ! $cmd
then
	echo "example_2 failed"
	exit 1
fi
echo "-------------------------------------------------------------------"
cmd="python example_3.py"
echo $cmd
if ! $cmd
then
	echo "example_3 failed"
	exit 1
fi
