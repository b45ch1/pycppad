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
echo "# Build documentation --------------------------------------------------"
yyyymmdd=`date +%G%m%d`
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
echo "# Run setup.py --------------------------------------------------" 
cmd="./example.setup.py clean --all"
echo "$cmd"
if ! $cmd
then
	echo "Cannot remove previous setup output."
	exit 1
fi
cmd="./example.setup.py build_ext --inplace --debug --undef NDEBUG"
echo "$cmd"
if ! $cmd
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
