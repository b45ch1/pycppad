# ifndef PYCPPAD_SETUP_INCLUDED
# define PYCPPAD_SETUP_INCLUDED

// Supress warnings about using deprecated numpy features:
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include <cppad/cppad.hpp>
# include <boost/python.hpp>
# include <numpy/arrayobject.h>
# include <numeric>
# include <iostream>
# include <string>
# include <cassert>
# include <exception>

# define PYCPPAD_ASSERT(expression, message) \
{ 	if( ! ( expression ) )                    \
        CppAD::ErrorHandler::Call(                \
 		(message[0] != '\0') ,            \
 		__LINE__             ,            \
 		__FILE__             ,            \
		#expression          ,            \
		message              );           }


namespace pycppad {
	typedef CppAD::AD<double>      AD_double;
	typedef CppAD::AD<AD_double>   AD_AD_double;

	using std::cout;
	using std::endl;
	using boost::python::handle;
	using boost::python::object;
	using boost::python::numeric::array;
	using boost::python::extract;

	class exception : public std::exception
	{	
	private : 
		char message_[201];
	public :
		exception(const char* message)
		{	strncpy(message_, message, 200);
			message_[200] = '\0';
		}
		const char* what(void) const throw()
		{	return message_; }
	};
}

# endif
