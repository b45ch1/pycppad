#! /bin/bash
if [ "$1" != "omhelp" ] && [ "$1" != "sdist" ] && [ "$1" != "all" ]
then
	echo "build.sh option, where option is one of the following"
	echo "omhelp: stop when the help is done"
	echo "sdist:  stop when the done building the source distribution"
	echo "all:    go all the way"
	exit 1
fi
option="$1"
# ---------------------------------------------------------------------
cppad_version="20080919.0"       # cppad release version we are using
yyyymmdd=`date +%G%m%d`          # todays year, month, and day
# ---------------------------------------------------------------------
# Note for Sebastian: this choice forces cppad to be dowloanded each time. 
# I like to use $HOME/install to speed up testing.
cppad_parent_dir="$HOME/install"             # directory for cppad tarball etc
# ---------------------------------------------------------------------
omhelp_location=`which omhelp`
if [ "$omhelp_location" = "" ]
then
	echo "Cannot find the omhelp command in your path"
	echo "skipping the build of the documentation"
fi
# ----------------------------------------------------------------------------
# Create setup.py from setup.template with certain replacements
sed < ./setup.template > setup.py \
	-e "s|\(package_version *=\).*|\1 '$yyyymmdd'|"  \
	-e "s|\(cppad_version *=\).*|\1 '$cppad_version'|" \
	-e "s|\(cppad_parent_dir *=\).*|\1 '$cppad_parent_dir'|"
chmod +x setup.py
# ----------------------------------------------------------------------------
if [ "$omhelp_location" != "" ]
then
	# check that every example file is documented
	for file in example/*.py
	do
		name=`echo $file | sed -e 's|example/||'`
		if ! grep "$name" omh/example.omh > /dev/null
		then
			echo "$file is not listed in omh/example.omh"
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
	if grep '^OMhelp Warning:' omhelp.log
	then
		echo "There are warnings in doc/omhelp.log"
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
	if [ "$option" == "omhelp" ]
	then
		exit 0
	fi
fi
# ----------------------------------------------------------------------------
# Create test_example.py
cat example/*.py > test_example.py
cat << EOF   >> test_example.py
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
    exit(0)
  else :
    print "%d tests failed" % number_fail 
    exit(1)
EOF
echo "# Create a source distribution ----------------------------------" 
cmd="rm -rf dist"
echo "$cmd"
$cmd
if [ -e "dist" ] 
then
	echo "Cannot remove previous source distribution."
	exit 1
fi
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
./setup.py sdist
if [ "$option" == "sdist" ]
then
	exit 0
fi
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
if [ ! -e pycppad/cppad_.so ]
then
	echo "setup.py failed to create pycppad/cppad_.so"
	exit 1
fi
# ----------------------------------------------------------------------------
if ! python test_example.py | tee test_example.out
then
	echo "test_example failed."
	exit 1
fi
number=`grep '^All' test_example.out | sed -e 's|All \([0-9]*\) .*|\1|'`
check=`grep '^def' test_example.py | wc -l`
if [ "$number" != "$check" ]
then
	echo "Expected $check tests but only found $number"
	exit 1
fi
echo
if ! python test_more.py | tee test_more.out
then
	echo "test_more.py failed."
	exit 1
fi
number=`grep '^All' test_more.out | sed -e 's|All \([0-9]*\) .*|\1|'`
check=`grep '^def' test_more.py | wc -l`
if [ "$number" != "$check" ] 
then
	echo "Expected $check tests but only found $number"
	exit 1
fi
echo
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
