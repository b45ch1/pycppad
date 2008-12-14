# ifndef PYTHON_CPPAD_SETUP_INCLUDED
# define PYTHON_CPPAD_SETUP_INCLUDED

# include <cppad/cppad.hpp>
# include <boost/python.hpp>
# include <numpy/noprefix.h>
# include <numeric>
# include <iostream>
# include <string>
# include <cassert>

# define PYTHON_CPPAD_ASSERT(expression, message) \
{ 	if( ! ( expression ) )                    \
        CppAD::ErrorHandler::Call(                \
 		(message[0] != '\0') ,            \
 		__LINE__             ,            \
 		__FILE__             ,            \
		#expression          ,            \
		message              );           }


namespace python_cppad {
	typedef CppAD::AD<double>      AD_double;
	typedef CppAD::AD<AD_double>   AD_AD_double;

	using std::cout;
	using std::endl;
	using boost::python::handle;
	using boost::python::object;
	using boost::python::numeric::array;
	using boost::python::extract;
}

# endif
