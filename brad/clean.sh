#! /bin/bash
list="
	libpython_cppad.so.1.0
	python_cppad.o
	python_cppad.tmp
	vector.o
	cppad/__init__.pyc
	cppad/python_cppad.so
"
for file in $list
do
	cmd="rm $file"
	echo $cmd
	$cmd
done
