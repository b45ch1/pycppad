# /bin/bash
set -e
# Modified version of http://www.boost.org/doc/libs/1_45_0/libs/python/doc/v2/
#	exception_translator.html
#
echo "cat << EOF > exception.cpp"
cat << EOF > exception.cpp
# include <boost/python/module.hpp>
# include <boost/python/def.hpp>
# include <boost/python/exception_translator.hpp>
# include <exception>
# include <string>

# if 0
struct my_exception : std::exception
{
  char const* what() throw() { return "One of my exceptions"; }
};
# else
class my_exception : public std::exception
{	
private : 
	char message_[201];
public :
	my_exception(const char* message)
	{	strncpy(message_, message, 200);
		message_[200] = '\0';
	}
	const char* what(void) const throw()
	{	return message_; }

};
# endif

void translate(my_exception const& e)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

void something_which_throws()
{
    throw my_exception("my error message");
}

BOOST_PYTHON_MODULE(exception)
{
  using namespace boost::python;
  register_exception_translator<my_exception>(&translate);
  
  def("something_which_throws", something_which_throws);
}
EOF
#
echo "gcc  -I/usr/include/python2.6 -g -c exception.cpp"
gcc  -I/usr/include/python2.6 -g -c exception.cpp 
#
echo "g++ -shared \\"
echo "	-Wl,--enable-auto-image-base \\"
echo "	-g \\"
echo "	exception.o \\"
echo "	-L/usr/lib -L/usr/lib/python2.6/config  \\"
echo "	-lboost_python-mt -lpython2.6 \\"
echo "	-o exception.dll"
g++ -shared \
	-Wl,--enable-auto-image-base \
	-g \
	exception.o \
	-L/usr/lib -L/usr/lib/python2.6/config \
	-lboost_python-mt -lpython2.6 \
	-o exception.dll
#
echo "cat << EOF > exception.py"
cat << EOF > exception.py
from sys import exit
import exception
try :
	exception.something_which_throws()
except RuntimeError :
	exit(0) 
exit(1)
EOF
if python exception.py
then
	flag=0
	echo "exception.sh: OK"
	for ext in .cpp .o .dll .py
	do
		echo "rm exception$ext"
		rm exception$ext
	done
else
	echo "exception.sh: Error"
	flag=1
fi
exit $flag
