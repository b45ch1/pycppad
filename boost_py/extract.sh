# /bin/bash
set -e
# Modified version of
# http://stackoverflow.com/questions/940132/
#	how-to-pass-pointer-to-an-array-in-python-for-a-wrapped-c-function
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
echo "cat << EOF > extract.cpp"
cat << EOF > extract.cpp
# include <boost/python.hpp>
# include <iostream>

namespace { // Avoid cluttering the global namespace.
	using std::cout;
	using boost::python::list;
	using boost::python::extract;

	void print_list(list list_obj)
	{	size_t n = len(list_obj);
		size_t i;
		cout << "[ ";
		if( n > 0 ) 
		{	for(i = 0; i < n-1; i++)
				cout << extract<double>( list_obj[i] ) << ", ";
			cout << extract<double>( list_obj[n-1] );
		}
		cout << " ]" << std::endl;
	}
}


BOOST_PYTHON_MODULE(extract)
{
    // Add regular function to the module.
    boost::python::def("print_list", print_list);
}
EOF
#
echo "gcc  -I/usr/include/python$python_version -g -c extract.cpp"
gcc  -I/usr/include/python$python_version -g -c extract.cpp 
#
echo "g++ -shared $extra_compile \\"
echo "	-g \\"
echo "	extract.o \\"
echo "	-L/usr/lib -L/usr/lib/python$python_version/config  \\"
echo "	-lboost_python-mt -lpython$python_version \\"
echo "	-o extract$library_extension"
g++ -shared $extra_compile \
	-g \
	extract.o \
	-L/usr/lib -L/usr/lib/python$python_version/config \
	-lboost_python-mt -lpython$python_version \
	-o extract$library_extension
#
echo "cat << EOF > extract.py"
cat << EOF > extract.py
import numpy
import extract
# mix integer and floats in list
list_obj = [1, 2., 3]
extract.print_list(list_obj)
EOF
#
if python extract.py
then
	flag=0
	echo "extract.sh: OK"
	for ext in .cpp .o $library_extension .py
	do
		echo "rm extract$ext"
		rm extract$ext
	done
else
	echo "extract.sh: Error"
	flag=1
fi
exit $flag
