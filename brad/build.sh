#! /bin/bash
# ---------------------------------------------------------------------
# User options
cppad_include_dir="/home/bradbell/CppAD/trunk"  # directory where CppAD is
boost_lib_dir="/usr/lib"         # directory where boost_python_lib is
boost_python_lib="boost_python"  # name of boost::python library 
# ---------------------------------------------------------------------
if [ ! -e $cppad_include_dir/cppad/cppad.hpp ]
then
	echo "Cannot find the CppAD include file cppad/cppad.hpp"
	echo "in the directory $cppad_include_dir."
	echo "Use the following web page for information about CppAD"
	echo "	http://www.coin-or.org/CppAD/"
	echo "Make sure that cppad_include_dir is set correctly"
	echo "at the beginning of the file ./build.sh"
	exit 1
fi
match=`ls $boost_lib_dir | grep "lib$boost_python_lib\."`
if [ "$match" == "" ]
then
	echo "Cannot find the boost::python library lib$boost_python_lib.*"
	echo "in the directory $boost_lib_dir."
	echo "Use the following web page for information about boost::python"
	echo "	http://www.boost.org/doc/libs/1_37_0/libs/python/doc/index.html"
	echo "Make sure that boost_lib_dir and boost_python_lib are set correctly"
	echo "at the beginnin of the file ./build.sh"
	exit 1
fi
#
location=`which omhelp`
if [ "$location" = "" ]
then
	echo "Cannot find the omhelp command in your path"
	echo "Use the following web page to download and install omhelp"
	echo "	http://www.seanet.com/~bradbell/omhelp/install.xml"
	exit 1
fi
location=`which py.test`
if [ "$location" == "" ]
then
	echo "Cannot find py.test in your path"
	echo "On ubuntu, the following command installs py.test"
	echo "	sudo apt-get install python-codespeak-lib"
	exit 1
fi
# ----------------------------------------------------------------------------
# Create setup.py with todays year, month, and day in yyyymmdd format
yyyymmdd=`date +%G%m%d`
sed < ./setup.template > setup.py \
	-e "s|\(package_version *=\).*|\1 '$yyyymmdd'|"  \
	-e "s|\(cppad_include_dir *=\).*|\1 '$cppad_include_dir'|" \
	-e "s|\(boost_lib_dir *=\).*|\1 '$boost_lib_dir'|" \
	-e "s|\(boost_python_lib *=\).*|\1 '$boost_python_lib'|"
chmod +x setup.py
echo "# Build documentation --------------------------------------------------"
sed -i doc.omh -e "s/pycppad-[0-9]{8}/pycppad-$yyyymmdd/"
if [ -e doc ]
then
	echo "rm -r doc"
	if ! rm -r doc
	then
		echo "Cannot remove old documentation directory"
		exit 1
	fi
fi
mkdir doc
cd doc
if ! omhelp ../doc.omh -xml -noframe -debug | tee omhelp.log
then
	echo "Error while building xml documentatioin"
	exit 1
fi
if ! omhelp ../doc.omh -xml -noframe -debug -printable
then
	echo "Error while building _printable.xml documentatioin"
	exit 1
fi
if ! omhelp ../doc.omh -noframe -debug 
then
	echo "Error while building html documentatioin"
	exit 1
fi
if ! omhelp ../doc.omh -noframe -debug -printable
then
	echo "Error while building _printable.html documentatioin"
	exit 1
fi
cd ..
echo "# Create a source distribution ----------------------------------" 
cmd="rm -rf dist"
echo "$cmd"
if ! $cmd
then
	echo "Cannot remove previous source distribution."
	exit 1
fi
cat << EOF > MANIFEST.in
include *.cpp
include *.hpp
include build.sh
include setup.py
include example/*
include doc.omh
include doc/*
include README
include test_more.py
EOF
./setup.py sdist
echo "# Extract the source distribution -------------------------------" 
cmd="cd dist"
echo "$cmd"
if ! $cmd
then
	echo "Cannot change into distribution directory."
	exit 1
fi
cmd="tar -xvzf pycppad-$yyyymmdd.tar.gz"
echo "$cmd"
if ! $cmd
then
	echo "Cannot extract the source distribution file"
	exit 1
fi
cmd="cd pycppad-$yyyymmdd"
if ! $cmd
then
	echo "Cannot change into the extracted soruce directory"
	exit 1
fi
echo "# Build the extension inplace -----------------------------------" 
cmd="./setup.py build_ext --inplace --debug --undef NDEBUG"
echo "$cmd"
# Kludge: setup.py is mistakenly putting -Wstrict-prototypes on compile line
$cmd 2>&1 |  sed -e '/warning: command line option "-Wstrict-prototypes"/d'
if [ ! -e cppad_.so ]
then
	echo "setup.py failed"
	exit 1
fi
# ----------------------------------------------------------------------------
echo 'from cppad import *' > test_example.py
cat example/*.py           >> test_example.py
if ! py.test test_example.py
then
	echo "test_example failed."
	exit 1
fi
if ! py.test test_more.py
then
	echo "test_more.py failed."
	exit 1
fi
echo "All tests passed."
check=`grep '^def' test_example.py | wc -l`
echo "Number of tests in test_example.py should be [$check]"
check=`grep '^def' test_more.py | wc -l`
echo "Number of tests in test_more.py should be [$check]"
# ----------------------------------------------------------------------------
dir="$HOME/prefix/pycppad"
cmd="rm -rf $dir"
echo "$cmd"
if ! $cmd
then
	echo "Cannot remove old version of $dir"
	exit 1
fi
cmd="./setup.py install --prefix=$HOME/prefix/pycppad"
echo "$cmd"
if ! $cmd
then
	echo "setup.py install failed"
	exit 1
fi
# ----------------------------------------------------------------------------
exit 0
