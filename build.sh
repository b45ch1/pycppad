#! /bin/bash -e
pycppad_version='20140710'
# ----------------------------------------------------------------------------
#
# exit on any error
set -e
#
if [ "$1" != "omhelp" ] &&  \
   [ "$1" != "sdist" ] &&  \
   [ "$1" != "all" ]
then
	echo "build.sh option, where option is one of the following"
	echo "omhelp: stop when the help is done"
	echo "sdist:  stop when the done building the source distribution"
	echo "all:    go all the way"
	exit 1
fi
option="$1"
# ---------------------------------------------------------------------
external_dir=`pwd`/external              # externals are placed here
log_dir=`pwd`                            # build progress logs written here
cppad_download_dir='http://www.coin-or.org/download/source/CppAD'
omhelp_download_dir='http://www.seanet.com/~bradbell'
# ----------------------------------------------------------------------------
# Update setup.py so it corresponds to current build.sh options above.
# only edit line corresponding to assignment statement, check not a == case
echo "sed < setup.py > setup.py.new"
sed < setup.py > setup.py.new \
	-e "s|\([\t ]*version *=\).*|\1 '$pycppad_version',|" 
if ! diff setup.py setup.py.new > /dev/null
then
	echo "Replacing setup.py using changes in setup.py.new"
	chmod +x setup.py.new
	mv setup.py.new setup.py
else
	echo "rm setup.py.new"
	rm setup.py.new
fi
# ----------------------------------------------------------------------------
if [ ! -d external ]
then
	echo "mkdir external"
	mkdir external
fi
cd external
if ! (ls | grep omhelp-) > /dev/null
then
	count="0"
else
	count=`ls | grep omhelp- | wc -l`
fi
if [ "$count" != "1" ]
then
	if [ "$count" != "0" ]
	then
		echo "rm -r omhelp-*"
		rm -r omhelp-*
	fi
	if [ -e OMhelp.unix.tar.gz ]
	then
		echo "rm OMhelp.unix.tar.gz"
		rm OMhelp.unix.tar.gz
	fi
	echo "curl -O $omhelp_download_dir/OMhelp.unix.tar.gz"
	curl -O "$omhelp_download_dir/OMhelp.unix.tar.gz"
	#
	echo "tar -xzf OMhelp.unix.tar.gz"
	tar -xzf OMhelp.unix.tar.gz
	#
	echo "cd omhelp-*"
	cd omhelp-* 
	#
	omhelp_dir=`pwd`
	#
	echo "./configure --prefix=$omhelp_dir/prefix"
	./configure --prefix="$omhelp_dir/prefix"
	#
	echo "make install"
	make install
	#
	cd ..
fi
count=`ls | grep omhelp-* | wc -l`
if [ "$count" != "1" ]
then
	echo "There should be one and only on omhelp-* directory in external"
	exit 1
fi
omhelp_dir=`ls | grep omhelp-`
cd ..
omhelp_dir="$external_dir/$omhelp_dir"
#
# check that every example file is documented
for file in example/*.py
do
	name=`echo $file | sed -e 's|example/||'`
	if ! grep "$name" omh/example.omh > /dev/null
	then
		echo "build.sh: $file is not listed in omh/example.omh"
		exit 1
	fi
done
#
# Change doc.omh and install.omh to use pycppad version in this file
for file in doc.omh omh/install.omh
do
	echo "sed < $file > $file.new"
	sed < $file > $file.new \
		-e "s/pycppad-[0-9]\{8\}/pycppad-$pycppad_version/"
	if ! diff $file $file.new > /dev/null
	then
		echo "Replacing $file using changes in $file.new"
		mv $file.new $file 
	else
		echo "rm $file.new"
		rm $file.new
	fi
done
#
if [ -e doc ]
then
	echo "rm -r doc"
	rm -r doc
fi
echo "mkdir doc"
mkdir doc
#
echo "cd doc"
cd doc
#
echo "omhelp ../doc.omh -xml -noframe -debug > omhelp.log"
$omhelp_dir/prefix/bin/omhelp \
	../doc.omh -xml -noframe -debug > $log_dir/omhelp.log
if grep '^OMhelp Warning:' $log_dir/omhelp.log
then
	echo "build.sh: There are warnings in omhelp.log"
	exit 1
fi
echo "omhelp ../doc.omh -xml -noframe -debug -printable > /dev/null"
$omhelp_dir/prefix/bin/omhelp \
	../doc.omh -xml -noframe -debug -printable > /dev/null
#
echo "omhelp ../doc.omh -noframe -debug > /dev/null"
$omhelp_dir/prefix/bin/omhelp \
	../doc.omh -noframe -debug > /dev/null
#
echo "omhelp ../doc.omh -noframe -debug -printable > /dev/null"
$omhelp_dir/prefix/bin/omhelp \
	../doc.omh -noframe -debug -printable > /dev/null
#
cd ..
if [ "$option" == "omhelp" ]
then
	exit 0
fi
# ----------------------------------------------------------------------------
for dir in dist pycppad-$pycppad_version
do
	if [ -e "$dir" ] 
	then
		echo "rm -r $dir"
		rm -r $dir
	fi
done
echo "create MANIFEST.in"
cat << EOF > MANIFEST.in
include pycppad/*.cpp
include pycppad/*.hpp
include build.sh
include setup.py
include example/*.py
include doc.omh
include doc/*
include README
include test_more.py
include test_example.py
EOF
echo "./setup.py sdist > setup.log"
./setup.py sdist > $log_dir/setup.log
if [ "$option" == "sdist" ]
then
	exit 0
fi
# ----------------------------------------------------------------------------
cmd="cd dist"
cd dist
#
echo "tar -xzf pycppad-$pycppad_version.tar.gz"
tar -xzf pycppad-$pycppad_version.tar.gz
#
echo "cd pycppad-$pycppad_version"
cd pycppad-$pycppad_version
#
echo "./setup.py build_ext --inplace --debug --undef NDEBUG >> setup.log"
./setup.py build_ext --inplace --debug --undef NDEBUG >> $log_dir/setup.log
#
if [ ! -e pycppad/cppad_.so ] && [ ! -e pycppad/cppad_.dll ]
then
	dir=`pwd`
	echo "build.sh: setup.py failed to create $dir/pycppad/cppad_.so"
	exit 1
fi
# ----------------------------------------------------------------------------
echo "python test_example.py > test_example.log"
python test_example.py > $log_dir/test_example.log
#
number=`grep '^All' $log_dir/test_example.log | \
	sed -e 's|All \([0-9]*\) .*|\1|'`
check=`cat example/*.py | grep '^def' | wc -l`
if [ "$number" != "$check" ]
then
	echo "build.sh: Expected $check tests but only found $number"
	exit 1
fi
echo "python test_more.py True > test_more.log"
python test_more.py True > $log_dir/test_more.log 
#
number=`grep '^All' $log_dir/test_more.log | \
	sed -e 's|All \([0-9]*\) .*|\1|'`
check=`grep '^def' test_more.py | wc -l`
if [ "$number" != "$check" ] 
then
	echo "build.sh: Expected $check tests but only found $number"
	exit 1
fi
#
dir=`pwd`
prefix_dir="$dir/prefix/pycppad"
#
echo "./setup.py install --prefix=$prefix_dir >> setup.log"
./setup.py install --prefix=$prefix_dir >> $log_dir/setup.log
#
if [ -e $prefix_dir/lib64 ]
then
	lib="lib64"
else
	lib="lib"
fi
python_version=`ls $prefix_dir/$lib`
if [ ! -d $prefix_dir/$lib/$python_version/site-packages/pycppad ] 
then
	echo "Install failed to create"
	echo "$prefix_dir/$python_version/site-packages/pycppad"
	exit 1
fi
# ----------------------------------------------------------------------------
if grep -i 'warning' $log_dir/setup.log
then
	echo "There are warnings in setup.log"
	exit 1
fi
echo "OK: build.sh $1"
exit 0
