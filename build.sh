#! /bin/bash
#
# exit on any error
set -e
#
if [ "$1" != "omhelp" ] &&  \
   [ "$1" != "sdist" ] &&  \
   [ "$1" != "all" ]   && \
   [ "$1" != "final" ] 
then
	echo "build.sh option, where option is one of the following"
	echo "omhelp: stop when the help is done"
	echo "sdist:  stop when the done building the source distribution"
	echo "all:    go all the way"
	echo "final:  go all the way and include download of cppad"
	exit 1
fi
option="$1"
# ---------------------------------------------------------------------
yyyymmdd=`date +%F | sed -e 's|-||g'`     # todays year, month, and day
cppad_tarball='cppad-20110101.2.gpl.tgz'  # name in download directory
cppad_parent_dir="$HOME/install"          # parrent of download directory
cppad_download_dir='http://www.coin-or.org/download/source/CppAD/'
log_dir=`pwd`
# ---------------------------------------------------------------------
# directory for cppad tarball (see setup.template)
cppad_dir=`echo $cppad_tarball | sed -e 's|\([^-]*-[0-9]\{8\}\.[0-9]*\).*|\1|'`
if [ "$option" == "final" ] && [ -e "$cppad_parent_dir/$cppad_dir" ]
then
	echo "rm -r $cppad_parent_dir/$cppad_dir"
	rm -r $cppad_parent_dir/$cppad_dir
fi
# ----------------------------------------------------------------------------
# Create setup.py from setup.template with certain replacements
# only edit line corresponding to assignment statement not check for ==
echo "sed < ./setup.template > setup.py -e ..."
sed < ./setup.template > setup.py \
	-e "s|\(package_version *=\)[^=].*|\1 '$yyyymmdd'|"  \
	-e "s|\(cppad_tarball *=\)[^=].*|\1 '$cppad_tarball'|" \
	-e "s|\(cppad_download_dir *=\)[^=].*|\1 '$cppad_download_dir'|" 
#
echo "chmod +x setup.py"
chmod +x setup.py
# ----------------------------------------------------------------------------
omhelp_location=`which omhelp`
if [ "$omhelp_location" = "" ]
then
	echo "build.sh: Cannot find the omhelp command in your path"
	echo "skipping the build of the documentation"
else
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
	# Change doc.omh and install.omh to use todays yyyymmdd 
	sed -i doc.omh -e "s/pycppad-[0-9]\{8\}/pycppad-$yyyymmdd/"
	sed -i omh/install.omh -e "s/pycppad-[0-9]\{8\}/pycppad-$yyyymmdd/"
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
	omhelp ../doc.omh -xml -noframe -debug > $log_dir/omhelp.log
	if grep '^OMhelp Warning:' $log_dir/omhelp.log
	then
		echo "build.sh: There are warnings in omhelp.log"
		exit 1
	fi
	echo "omhelp ../doc.omh -xml -noframe -debug -printable > /dev/null"
	omhelp ../doc.omh -xml -noframe -debug -printable > /dev/null
	#
	echo "omhelp ../doc.omh -noframe -debug > /dev/null"
	omhelp ../doc.omh -noframe -debug > /dev/null
	#
	echo "omhelp ../doc.omh -noframe -debug -printable > /dev/null"
	omhelp ../doc.omh -noframe -debug -printable > /dev/null
	#
	cd ..
	if [ "$option" == "omhelp" ]
	then
		exit 0
	fi
fi
# ----------------------------------------------------------------------------
echo "Create test_example.py"
echo "#!/usr/bin/env python" > test_example.py
cat example/*.py >> test_example.py
cat << EOF   >> test_example.py
import sys
if __name__ == "__main__" :
  number_ok   = 0
  number_fail = 0
  list_of_globals = sorted( globals().copy() )
  for g in list_of_globals :
    if g[:13] == "pycppad_test_" :
      ok = True
      try :
        eval("%s()" % g)
      except AssertionError :
        ok = False
      if ok : 
        print "OK:    %s" % g[13:]
        number_ok = number_ok + 1
      else : 
        print "Error: %s" % g[13:]
        number_fail = number_fail + 1
  if number_fail == 0 : 
    print "All %d tests passed" % number_ok
    sys.exit(0)
  else :
    print "%d tests failed" % number_fail 
    sys.exit(1)
EOF
echo "chmod +x test_example.py"
chmod +x test_example.py
# ----------------------------------------------------------------------------
for dir in dist pycppad-$yyyymmdd
do
	if [ -e "$dir" ] 
	then
		echo "rm -rf $dir"
		rm -rf $dir
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
echo "tar -xzf pycppad-$yyyymmdd.tar.gz"
tar -xzf pycppad-$yyyymmdd.tar.gz
#
echo "cd pycppad-$yyyymmdd"
cd pycppad-$yyyymmdd
#
echo "./setup.py build_ext --inplace --debug --undef NDEBUG >> setup.log"
./setup.py build_ext --inplace --debug --undef NDEBUG >> $log_dir/setup.log
# Kludge: setup.py is mistakenly putting -Wstrict-prototypes on compile line
echo "sed -i $log_dir/setup.log \\"
echo "	-e '/warning: command line option \"-Wstrict-prototypes\"/d'"
sed -i $log_dir/setup.log \
	-e '/warning: command line option "-Wstrict-prototypes"/d'
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
check=`grep '^def' test_example.py | wc -l`
if [ "$number" != "$check" ]
then
	echo "build.sh: Expected $check tests but only found $number"
	exit 1
fi
echo "python test_more.py > test_more.log"
python test_more.py > $log_dir/test_more.log 
#
number=`grep '^All' $log_dir/test_more.log | \
	sed -e 's|All \([0-9]*\) .*|\1|'`
check=`grep '^def' test_more.py | wc -l`
if [ "$number" != "$check" ] 
then
	echo "build.sh: Expected $check tests but only found $number"
	exit 1
fi
# ----------------------------------------------------------------------------
echo "rm -rf $HOME/prefix/pycppad"
rm -rf $HOME/prefix/pycppad
#
echo "./setup.py install --prefix=$HOME/prefix/pycppad >> setup.log"
./setup.py install --prefix=$HOME/prefix/pycppad >> $log_dir/setup.log
# ----------------------------------------------------------------------------
echo "OK: build.sh $1"
exit 0
