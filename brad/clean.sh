#! /bin/bash
list="
	libpycppad.so.1.0
	pycppad.o
	pycppad.tmp
	vector.o
	cppad/__init__.pyc
	cppad/pycppad.so
"
for file in $list
do
	cmd="rm $file"
	echo $cmd
	$cmd
done
