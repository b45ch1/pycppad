# /bin/bash
set -e
# Modified version of
# http://wiki.python.org/moin/boost.python/SimpleExample
#
echo "cat << EOF > boost_py_simple.cpp"
cat << EOF > boost_py_simple.cpp
# include <string>

namespace { // Avoid cluttering the global namespace.
  int square(int number) { return number * number; }
}

# include <boost/python.hpp>

BOOST_PYTHON_MODULE(boost_py_simple)
{
    // Add regular function to the module.
    boost::python::def("square", square);
}
EOF
#
echo "gcc  -I/usr/include/python2.6 -g -c boost_py_simple.cpp"
gcc  -I/usr/include/python2.6 -g -c boost_py_simple.cpp 
#
echo "g++ -shared \\"
echo "	-Wl,--enable-auto-image-base \\"
echo "	-g \\"
echo "	boost_py_simple.o \\"
echo "	-L/usr/lib -L/usr/lib/python2.6/config  \\"
echo "	-lboost_python-mt -lpython2.6 \\"
echo "	-o boost_py_simple.dll"
g++ -shared \
	-Wl,--enable-auto-image-base \
	-g \
	boost_py_simple.o \
	-L/usr/lib -L/usr/lib/python2.6/config \
	-lboost_python-mt -lpython2.6 \
	-o boost_py_simple.dll
#
echo "cat << EOF > boost_py_simple.py"
cat << EOF > boost_py_simple.py
import boost_py_simple
number = 11
ok     = number * number == boost_py_simple.square(number)
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
if python boost_py_simple.py
then
	flag=0
	echo "boost_py_simple.sh: OK"
	for ext in .cpp .o .dll .py
	do
		echo "rm boost_py_simple$ext"
		rm boost_py_simple$ext
	done
else
	echo "boost_py_simple.sh: Error"
	flag=1
fi
exit $flag
