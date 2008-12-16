#include "py_cppad.hpp"

namespace{

	/* ERROR HANDLER */
	void python_cppad_error_handler(bool known,	int  line, const char *file, const char *exp, const char *msg){
		if( ! known ) msg =
			"Bug detected in python_cppad, Please report this.";
		PyErr_SetString(PyExc_ValueError, msg);
		bp::throw_error_already_set();
	}

	// This ojbect lasts forever, so forever replacement of
	// the default CppAD erorr handler
	CppAD::ErrorHandler myhandler(python_cppad_error_handler);

	/* =================================== */
	/* FUNCTIONS                           */
	/* =================================== */

	void Independent(bpn::array& x_array, int level){
		if( level == 1){
			AD_double_vec x_vec(x_array);
			CppAD::Independent(x_vec);
		}
		else if(level == 2){
			AD_AD_double_vec x_vec(x_array);
			CppAD::Independent(x_vec);
		}
		else{
			CppAD::ErrorHandler::Call(1, __LINE__, __FILE__, "Independent(array& x_array, int level)\n", "This level is not supported!\n" );
		}
	}
	
}
