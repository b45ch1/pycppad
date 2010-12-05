# /bin/bash
set -e
# Modified version of
# http://wiki.python.org/moin/boost.python/SimpleExample
# ----------------------------------------------------------------------
python_version=`ls /usr/include | grep python | sed -e 's/python//'` 
system=`uname | sed -e 's/\(......\).*/\1/'`
if [ "$system" == "CYGWIN" ]
then
	extra_compile="-Wl,--enable-auto-image-base"
	library_extension=".dll"
else
	extra_compile=""
	library_extension=".so"
fi
# ----------------------------------------------------------------------
echo "cat << EOF > simple.cpp"
cat << EOF > simple.cpp
# include <string>

namespace { // Avoid cluttering the global namespace.
  int square(int number) { return number * number; }
}

# include <boost/python.hpp>

BOOST_PYTHON_MODULE(simple)
{
    // Add regular function to the module.
    boost::python::def("square", square);
}
EOF
#
echo "gcc  -I/usr/include/python$python_version -g -c simple.cpp"
gcc  -I/usr/include/python$python_version -g -c simple.cpp 
#
echo "g++ -shared $extra_compile \\"
echo "	-g \\"
echo "	simple.o \\"
echo "	-L/usr/lib -L/usr/lib/python$python_version/config  \\"
echo "	-lboost_python-mt -lpython$python_version \\"
echo "	-o simple$library_extension"
g++ -shared $extra_compile \
	-g \
	simple.o \
	-L/usr/lib -L/usr/lib/python$python_version/config \
	-lboost_python-mt -lpython$python_version \
	-o simple$library_extension
#
echo "cat << EOF > simple.py"
cat << EOF > simple.py
import simple
number = 11
ok     = number * number == simple.square(number)
#
# use sys to get exit function
import sys
if ok :
	# ok case so return non error flag
	sys.exit(0)
else :
	# error case so return with error flag set
	sys.exit(1)
EOF
#
if python simple.py
then
	flag=0
	echo "simple.sh: OK"
	for ext in .cpp .o $library_extension .py
	do
		echo "rm simple$ext"
		rm simple$ext
	done
else
	echo "simple.sh: Error"
	flag=1
fi
exit $flag
