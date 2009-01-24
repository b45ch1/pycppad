#! /bin/bash
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
sed < ./example.setup.py > setup.py \
	-e "s/\(package_version *=\).*/\1 '$yyyymmdd'/" 
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
# Kludge: move debug and other local setting to this script 
# from example.setup.py
cmd="./setup.py build_ext --inplace --debug"
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
exit 0
