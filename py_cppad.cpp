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
	
	void Independent(bpn::array& x_array){
		AD_double_vec x_vec(x_array);
		CppAD::Independent(x_vec);
	}

	/* =================================== */
	/* CLASS: ADFun                        */
	/* =================================== */

	ADFun_double::ADFun_double(bpn::array& x_array, bpn::array& y_array){
		AD_double_vec x_vec(x_array);
		AD_double_vec y_vec(y_array);
		f_.Dependent(x_vec, y_vec);
	}

	bpn::array ADFun_double::Forward(int p, bpn::array& xp){
	 	size_t     p_sz(p);
		double_vec xp_vec(xp);
		double_vec result = f_.Forward(p_sz, xp_vec);
		return vector2array(result);
	}

	bpn::array vector2array(const vec<double>& in_vec){
		int n = static_cast<int>( in_vec.size() );
		assert( n >= 0 );

		bp::object obj(bp::handle<>( PyArray_FromDims(1, &n, PyArray_DOUBLE) ));
		// for some unknown reason,
		// static_cast<PyArrayObject*> ( obj.ptr() ) does not work ?
		double *ptr = static_cast<double*> ( PyArray_DATA (
			(PyArrayObject*) ( obj.ptr() )
		));
		for(size_t i = 0; i < in_vec.size(); i++){
			ptr[i] = in_vec[i];
		}
		return  static_cast<bpn::array>( obj );
	}

}
