#ifndef PY_CPPAD_HPP
#define PY_CPPAD_HPP

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include <string>
#include <boost/python.hpp>
#include <numpy/noprefix.h>


#include "cppad/cppad.hpp"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;



namespace{

	/* =================================== */
	/* ERROR HANDLER                       */
	/* =================================== */

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
	/* CLASS VEC<DOUBLE>                   */
	/* =================================== */

	template<class Tdouble>
	class vec {
			size_t       length_;
			Tdouble  *pointer_;
			Tdouble **handle_;
			bool    allocated_;
		public:
			typedef Tdouble value_type;
			vec(bpn::array& py_array);
			vec(size_t length);
			vec(const vec<Tdouble>& in_vec);
			vec(void);
			~vec(void);
			void operator=(const vec<Tdouble>& in_vec);
			size_t size(void) const;
			void resize(size_t length);
			Tdouble& operator[](size_t i);
			const Tdouble& operator[](size_t i) const;
	};

	template<>
	const double& vec<double>::operator[](size_t i) const{
		assert( i < length_ );
		return pointer_[i];
	}
	
	template<>
	vec<double>::vec(bpn::array& py_array)
	{	// get array info
		int* dims_ptr = PyArray_DIMS(py_array.ptr());
// 		int ndim      = PyArray_NDIM(py_array.ptr());
		int length    = dims_ptr[0];

		// check array info
		assert( ndim == 1 );
		assert( length >= 0 );

		// set private data
		length_    = static_cast<size_t>( length );
		pointer_   = static_cast<double*>(
			PyArray_DATA(py_array.ptr())
		);
		allocated_ = false;
		return;
	}
	
	template<>
	vec<double>::vec(size_t length){
		length_    = length;
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		allocated_ = true;
		return;
	}
	template<>
	vec<double>::vec(const vec<double>& vec)
	{	length_    = vec.length_;
		pointer_   = CPPAD_TRACK_NEW_VEC(length_, pointer_);
		allocated_ = true;
		for(size_t i = 0; i < length_; i++)
			pointer_[i] = vec[i];
	}
	template<>
	vec<double>::vec(void){
		length_    = 0;
		pointer_   = 0;
		allocated_ = false;
	}
	
	template<>
	vec<double>::~vec(void){
		if( allocated_ )
			CPPAD_TRACK_DEL_VEC(pointer_);
	}

	template<>
	void vec<double>::operator=(const vec<double>& vec){
		assert( length_ == vec.length_ );
		for(size_t i = 0; i < length_; i++){
			pointer_[i] = vec.pointer_[i];
		}
	}

	template<>
	size_t vec<double>::size(void) const{
		return length_;
	}

	template<>
	void vec<double>::resize(size_t length){
		if( allocated_ ){
			CPPAD_TRACK_DEL_VEC(pointer_);
		}
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		length_    = length;
		allocated_ = true;
	}

	template<>
	double& vec<double>::operator[](size_t i){
		assert( i < length_ );
		return pointer_[i];
	}

	/* =================================== */
	/* CLASS vec<Tdouble>                  */
	/* =================================== */

	template<class Tdouble>
	vec<Tdouble>::vec(bpn::array& py_array){
		// get array info
		int* dims_ptr = PyArray_DIMS(py_array.ptr());
// 		int ndim      = PyArray_NDIM(py_array.ptr());
		int length    = dims_ptr[0];

		// check array info
		assert( ndim == 1 );
		assert( length >= 0 );

		// pointer to object
		bp::object *obj_ptr = static_cast<bp::object*>(
			PyArray_DATA(py_array.ptr())
		);

		// set private data
		length_  = static_cast<size_t>(length);
		pointer_ = 0;
		handle_  = CPPAD_TRACK_NEW_VEC(length_, handle_);
		for(size_t i = 0; i < length_; i++){
			handle_[i] = & bp::extract<Tdouble&>(obj_ptr[i])();
		}
		return;
	}

	template<class Tdouble>
	vec<Tdouble>::vec(size_t length){
		length_  = length;
		pointer_ = CPPAD_TRACK_NEW_VEC(length, pointer_);
		handle_  = CPPAD_TRACK_NEW_VEC(length, handle_);
		for(size_t i = 0; i < length_; i++)
			handle_[i] = pointer_ + i;
		return;
	}

	template<class Tdouble>
	vec<Tdouble>::vec(const vec<Tdouble>& in_vec)	{
		length_   = in_vec.length_;
		pointer_  = CPPAD_TRACK_NEW_VEC(length_, pointer_);
		handle_   = CPPAD_TRACK_NEW_VEC(length_, handle_);
		for(size_t i = 0; i < length_; i++)
		{	handle_[i]  = pointer_ + i;
			pointer_[i] = in_vec[i];
		}
	}

	template<class Tdouble>
	vec<Tdouble>::vec(void){
		length_  = 0;
		pointer_ = 0;
		handle_  = 0;
	}

	template<class Tdouble>
	vec<Tdouble>::~vec(void){
		if( handle_ != 0 )
			CPPAD_TRACK_DEL_VEC(handle_);
		if( pointer_ != 0 )
			CPPAD_TRACK_DEL_VEC(pointer_);
	}

	template<class Tdouble>
	void vec<Tdouble>::operator=(const vec<Tdouble>& in_vec)	{
		assert( length_ == in_vec.length_ );
		for(size_t i = 0; i < length_; i++)
			*handle_[i] = *(in_vec.handle_[i]);
		return;
	}

	template<class Tdouble>
	size_t vec<Tdouble>::size(void) const{
		return length_;
	}

	template<class Tdouble>
	void vec<Tdouble>::resize(size_t length)
	{
		if( handle_ != 0 )
			CPPAD_TRACK_DEL_VEC(handle_);
		if( pointer_ != 0 )
			CPPAD_TRACK_DEL_VEC(pointer_);
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		handle_    = CPPAD_TRACK_NEW_VEC(length, handle_);
		length_    = length;
		for(size_t i = 0; i < length_; i++)
			handle_[i]  = pointer_ + i;
	}

	template<class Tdouble>
	Tdouble& vec<Tdouble>::operator[](size_t i){
		assert( i < length_ );
		return *handle_[i];
	}

	template<class Tdouble>
	const Tdouble& vec<Tdouble>::operator[](size_t i) const{
		assert( i < length_ );
		return *handle_[i];
	}




	/* =================================== */
	/* HELPER FUNCTIONOS                   */
	/* =================================== */
	template<class Tdouble>
	bpn::array vector2array(const vec<Tdouble> &in_vec);

	template<>
	bpn::array vector2array<double>(const vec<double>& in_vec){
		int n = static_cast<int>( in_vec.size() );
		assert( n >= 0 );

		bp::object obj(bp::handle<>( PyArray_FromDims(1, &n, PyArray_DOUBLE) ));
		double *ptr = static_cast<double*> ( PyArray_DATA (
			reinterpret_cast<PyArrayObject*> ( obj.ptr() )
		));
		for(size_t i = 0; i < in_vec.size(); i++){
			ptr[i] = in_vec[i];
		}
		return  static_cast<bpn::array>( obj );
	}

	template<class Tdouble>
	bpn::array vector2array(const vec<Tdouble>& in_vec){
		int n = static_cast<int>( in_vec.size() );
		assert( n >= 0 );
		boost::python::object retval(boost::python::handle<>( PyArray_FromDims(1, &n, PyArray_OBJECT) ));

		for(size_t i = 0; i < in_vec.size(); i++){
			retval[i] = in_vec[i];
		}
		return  static_cast<bpn::array>( retval );
	}




	/* =================================== */
	/* CLASS  ADFUN                        */
	/* =================================== */

	template<class Tdouble>
	class ADFun {
		private:
			CppAD::ADFun<Tdouble> f_;
		public:
			ADFun(bpn::array& x_array, bpn::array& y_array);
			bpn::array Forward(int N, bpn::array& xd);
			bpn::array Reverse(int M, bpn::array& wd);
			bpn::array Jacobian(bpn::array& x_array);
			bpn::array Hessian(bpn::array& x_array, bpn::array& w_array);
	};

	template<class Tdouble>
	ADFun<Tdouble>::ADFun(bpn::array& x_array, bpn::array& y_array){
		vec<CppAD::AD<Tdouble> > x_vec(x_array);
		vec<CppAD::AD<Tdouble> > y_vec(y_array);
		f_.Dependent(x_vec, y_vec);
	}

	template<class Tdouble>
	bpn::array ADFun<Tdouble>::Forward(int p, bpn::array& xp){
	 	size_t     p_sz(p);
		vec<Tdouble> xp_vec(xp);
		vec<Tdouble> result = f_.Forward(p_sz, xp_vec);
		return vector2array(result);
	}

	template<class Tdouble>
	bpn::array ADFun<Tdouble>::Reverse(int M, bpn::array& wd){
	 	size_t     M_sz(M);
		vec<Tdouble> wd_vec(wd);
		vec<Tdouble> result = f_.Reverse(M_sz, wd_vec);
		return vector2array(result);
	}

	template<class Tdouble>
	bpn::array ADFun<Tdouble>::Jacobian(bpn::array& x_array){
		vec<Tdouble> x_vec(x_array);
		vec<Tdouble> result = f_.Jacobian(x_vec);

		int N = f_.Domain();
		int M = f_.Range();

		int dimens[2];
		dimens[0] = M;
		dimens[1] = N;
		bp::object obj(bp::handle<>( PyArray_FromDims(2, dimens, PyArray_DOUBLE)));
		double *ptr = static_cast<double*> ( PyArray_DATA (
			reinterpret_cast<PyArrayObject*> ( obj.ptr() )
		));
		for(int m = 0; m < M; ++m){
			for(int n = 0; n < N; ++n){
				ptr[m*N + n] = result[m*N + n];
			}
		}
		return  static_cast<bpn::array>( obj );
	}

	template<class Tdouble>
	bpn::array ADFun<Tdouble>::Hessian(bpn::array& x_array, bpn::array& w_array){
		vec<Tdouble> x_vec(x_array);
		vec<Tdouble> w_vec(w_array);

		vec<Tdouble> result = f_.Hessian(x_vec, w_vec);

		int N = f_.Domain();
		int M = N;

		int dimens[2];
		dimens[0] = M;
		dimens[1] = N;
		bp::object obj(bp::handle<>( PyArray_FromDims(2, dimens, PyArray_DOUBLE)));
		double *ptr = static_cast<double*> ( PyArray_DATA (
			reinterpret_cast<PyArrayObject*> ( obj.ptr() )
		));
		for(int m = 0; m < M; ++m){
			for(int n = 0; n < N; ++n){
				ptr[m*N + n] = result[m*N + n];
			}
		}
		return  static_cast<bpn::array>( obj );
	}
	



	/* =================================== */
	/* GENERAL FUNCTIONS                   */
	/* =================================== */

	typedef CppAD::AD<double> AD_double;
	typedef CppAD::AD<CppAD::AD<double> > AD_AD_double;
	typedef vec<double> double_vec;
	typedef vec<AD_double> AD_double_vec;
	typedef vec<AD_AD_double> AD_AD_double_vec;
	typedef ADFun<double> ADFun_double;
	typedef ADFun<AD_double> ADFun_AD_double;

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


	#define PYCPPAD_STD_MATH(Name, Base)                             \
	CppAD::AD<Base>	(* Name##_AD_##Base)( const CppAD::AD<Base>& ) = &CppAD::Name


	PYCPPAD_STD_MATH(asin , double);
	PYCPPAD_STD_MATH(acos , double);
	PYCPPAD_STD_MATH(atan , double);
	PYCPPAD_STD_MATH(cos  , double);
	PYCPPAD_STD_MATH(cosh , double);
	PYCPPAD_STD_MATH(exp  , double);
	PYCPPAD_STD_MATH(log  , double);
	PYCPPAD_STD_MATH(log10, double);
	PYCPPAD_STD_MATH(sin  , double);
	PYCPPAD_STD_MATH(sinh , double);
	PYCPPAD_STD_MATH(sqrt , double);
	PYCPPAD_STD_MATH(tan  , double);
	PYCPPAD_STD_MATH(tanh , double);
	
	PYCPPAD_STD_MATH(asin , AD_double);
	PYCPPAD_STD_MATH(acos , AD_double);
	PYCPPAD_STD_MATH(atan , AD_double);
	PYCPPAD_STD_MATH(cos  , AD_double);
	PYCPPAD_STD_MATH(cosh , AD_double);
	PYCPPAD_STD_MATH(exp  , AD_double);
	PYCPPAD_STD_MATH(log  , AD_double);
	PYCPPAD_STD_MATH(log10, AD_double);
	PYCPPAD_STD_MATH(sin  , AD_double);
	PYCPPAD_STD_MATH(sinh , AD_double);
	PYCPPAD_STD_MATH(sqrt , AD_double);
	PYCPPAD_STD_MATH(tan  , AD_double);
	PYCPPAD_STD_MATH(tanh , AD_double);

	
// 	/* atomic (aka elementary) operations */
// 	CppAD::AD<double>	(*cos_AD_double) 		( const CppAD::AD<double> & ) = &CppAD::cos;
// 	CppAD::AD<double>	(*sin_AD_double) 		( const CppAD::AD<double> & ) = &CppAD::sin;
// 
// 	CppAD::AD<AD_double>	(*sin_AD_AD_double) ( const CppAD::AD<AD_double> & ) = &CppAD::sin;
// 	CppAD::AD<AD_double>	(*cos_AD_AD_double) ( const CppAD::AD<AD_double> & ) = &CppAD::cos;



}

BOOST_PYTHON_MODULE(_cppad)
{
	using namespace boost::python;
	import_array(); 										/* some kind of hack to get numpy working */
	bpn::array::set_module_and_type("numpy", "ndarray");	/* some kind of hack to get numpy working */
	
	scope().attr("__doc__") =" CppAD: docstring \n\
							   next line";

	def("Independent", &Independent);


	# define PYCPPAD_BINARY(op)       \
	.def(self     op self)         \
	.def(double() op self)         \
	.def(self     op double())

	# define PYCPPAD_UNARY(op,Base)       \
	.def("op",     op##_AD_##Base)

	
	# define PYCPPAD_OPERATOR_LIST(Base)    \
	PYCPPAD_BINARY(+)         \
	PYCPPAD_BINARY(-)         \
	PYCPPAD_BINARY(*)         \
	PYCPPAD_BINARY(/)         \
	PYCPPAD_BINARY(<)         \
	PYCPPAD_BINARY(>)         \
	PYCPPAD_BINARY(<=)        \
	PYCPPAD_BINARY(>=)        \
	PYCPPAD_BINARY(==)        \
	PYCPPAD_BINARY(!=)        \
	.def(self += self)             \
	.def(self -= self)             \
	.def(self *= self)             \
	.def(self /= self)             \
	.def(self += double())         \
	.def(self -= double())         \
	.def(self *= double())         \
	.def(self /= double())        \
                                  \
	PYCPPAD_UNARY(asin,Base)      \
	PYCPPAD_UNARY(acos,Base)      \
	PYCPPAD_UNARY(atan,Base)      \
	PYCPPAD_UNARY(cos,Base)      \
	PYCPPAD_UNARY(cosh,Base)      \
	PYCPPAD_UNARY(exp,Base)      \
	PYCPPAD_UNARY(log,Base)      \
	PYCPPAD_UNARY(log10,Base)      \
	PYCPPAD_UNARY(sin,Base)      \
	PYCPPAD_UNARY(sinh,Base)      \
	PYCPPAD_UNARY(sqrt,Base)      \
	PYCPPAD_UNARY(tan,Base)      \
	PYCPPAD_UNARY(tanh,Base)


	class_<AD_double>("a_double", init<double>())
		.def(boost::python::self_ns::str(self))
		.add_property("value", &AD_double::value_)
		.add_property("id", &AD_double::id_)
		.add_property("taddr", &AD_double::taddr_)
		PYCPPAD_OPERATOR_LIST(double)
// 		.def("cos", cos_AD_double  )
// 		.def("sin", sin_AD_double  )
	;

	class_<AD_AD_double>("a2double", init<AD_double>())
		.def(boost::python::self_ns::str(self))
		.add_property("value", &AD_AD_double::value_)
		.add_property("id", &AD_AD_double::id_)
		.add_property("taddr", &AD_AD_double::taddr_)
		PYCPPAD_OPERATOR_LIST(AD_double)
// 		.def("cos", cos_AD_AD_double  )
// 		.def("sin", sin_AD_AD_double  )
	;
		
	class_<ADFun_double>("ADFun_double", init< bpn::array& , bpn::array& >())
		.def("forward", &ADFun_double::Forward)
		.def("reverse", &ADFun_double::Reverse)
		.def("jacobian",  &ADFun_double::Jacobian)
		.def("lagrange_hessian",  &ADFun_double::Hessian)
	;

	class_<ADFun_AD_double>("ADFun_AD_double", init< bpn::array& , bpn::array& >())
		.def("forward", &ADFun_AD_double::Forward)
		.def("reverse", &ADFun_AD_double::Reverse)
	;
}

#endif
