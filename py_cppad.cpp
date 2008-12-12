#include "py_cppad.hpp"

namespace{
/* operators */
// AD_double *AD_double_mul_AD_double_AD_double(const AD_double &lhs, const AD_double &rhs){
// 	return new AD_double(CppAD::operator*(lhs,rhs));
// }

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


	/* ADFun */
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

	/* functions */
	void Independent(bpn::array& x_array){
		AD_double_vec x_vec(x_array);
		CppAD::Independent(x_vec);
		return;
	}

	AD_double_vec::AD_double_vec(bpn::array& py_array)
	{
		// get array info
		int* dims_ptr = PyArray_DIMS(py_array.ptr());
		int ndim      = PyArray_NDIM(py_array.ptr());
		int length    = dims_ptr[0];

		// check array info
		assert( ndim == 1 );
		assert( length >= 0 );

		// pointer to object
		bp::object *obj_ptr = static_cast<bp::object*>( 
			PyArray_DATA(py_array.ptr()) 
		);

		// set private data
		using boost::python::extract;
		length_  = static_cast<size_t>(length);
		pointer_ = 0;
		handle_  = CPPAD_TRACK_NEW_VEC(length_, handle_);
		for(size_t i = 0; i < length_; i++) handle_[i] = 
			& extract<AD_double&>(obj_ptr[i])(); 
		return;
	}
	// constructor from size
	AD_double_vec::AD_double_vec(size_t length)
	{
		length_  = length;
		pointer_ = CPPAD_TRACK_NEW_VEC(length, pointer_);
		handle_  = CPPAD_TRACK_NEW_VEC(length, handle_);
		for(size_t i = 0; i < length_; i++)
			handle_[i] = pointer_ + i;
		return;
	}

	// copy constructor
	AD_double_vec::AD_double_vec(const AD_double_vec& vec)
	{
		length_   = vec.length_;
		pointer_  = CPPAD_TRACK_NEW_VEC(length_, pointer_);
		handle_   = CPPAD_TRACK_NEW_VEC(length_, handle_);
		for(size_t i = 0; i < length_; i++)
		{	handle_[i]  = pointer_ + i;
			pointer_[i] = vec[i];
		}
	}

	// default constructor
	AD_double_vec::AD_double_vec(void)
	{
		length_  = 0;
		pointer_ = 0;
		handle_  = 0;
	}

	// destructor
	AD_double_vec::~AD_double_vec(void)
	{
		if( handle_ != 0 )
			CPPAD_TRACK_DEL_VEC(handle_); 
		if( pointer_ != 0 )
			CPPAD_TRACK_DEL_VEC(pointer_);	
	}

	// assignment operator
	void AD_double_vec::operator=(const AD_double_vec& vec)
	{
		assert( length_ == vec.length_ ); 
		for(size_t i = 0; i < length_; i++)
			*handle_[i] = *(vec.handle_[i]);
		return;
	}

	// size member function
	size_t AD_double_vec::size(void) const
	{	return length_; }

	// resize 
	void AD_double_vec::resize(size_t length)
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

	// non constant element access
	AD_double& AD_double_vec::operator[](size_t i)
	{	assert( i < length_ );
		return *handle_[i];
	}

	// constant element access
	const AD_double& AD_double_vec::operator[](size_t i) const
	{	assert( i < length_ );
		return *handle_[i];
	}
	// -------------------------------------------------------------
	// class double_vec
	//
	// constructor from a python array
	double_vec::double_vec(bpn::array& py_array)
	{	// get array info
		int* dims_ptr = PyArray_DIMS(py_array.ptr());
		int ndim      = PyArray_NDIM(py_array.ptr());
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

	// constructor from size
	double_vec::double_vec(size_t length)
	{	// set private data
		length_    = length;
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		allocated_ = true;
		return;
	}

	// copy constructor
	double_vec::double_vec(const double_vec& vec)
	{	length_    = vec.length_;
		pointer_   = CPPAD_TRACK_NEW_VEC(length_, pointer_);
		allocated_ = true;
		for(size_t i = 0; i < length_; i++)
			pointer_[i] = vec[i];
	}

	// default constructor
	double_vec::double_vec(void)
	{	length_    = 0;
		pointer_   = 0;
		allocated_ = false;
	}

	// destructor
	double_vec::~double_vec(void)
	{	if( allocated_ )
			CPPAD_TRACK_DEL_VEC(pointer_);	
	}

	// assignment operator
	void double_vec::operator=(const double_vec& vec)
	{	assert( length_ == vec.length_ ); 
		for(size_t i = 0; i < length_; i++)
			pointer_[i] = vec.pointer_[i];
		return;
	}

	// size member function
	size_t double_vec::size(void) const
	{	return length_; }

	// resize 
	void double_vec::resize(size_t length)
	{	if( allocated_ )
			CPPAD_TRACK_DEL_VEC(pointer_);
		pointer_   = CPPAD_TRACK_NEW_VEC(length, pointer_);
		length_    = length;
		allocated_ = true;
	}

	// non constant element access
	double& double_vec::operator[](size_t i)
	{	assert( i < length_ );
		return pointer_[i];
	}

	// constant element access
	const double& double_vec::operator[](size_t i) const
	{	assert( i < length_ );
		return pointer_[i];
	}
	// -------------------------------------------------------------
	bpn::array vector2array(const double_vec& vec)
	{	int n = static_cast<int>( vec.size() );
		assert( n >= 0 );

		bp::object obj(bp::handle<>( PyArray_FromDims(1, &n, PyArray_DOUBLE) ));
		// for some unknown reason,
		// static_cast<PyArrayObject*> ( obj.ptr() ) does not work ?
		double *ptr = static_cast<double*> ( PyArray_DATA (
			(PyArrayObject*) ( obj.ptr() )
		));
		for(size_t i = 0; i < vec.size(); i++){
			ptr[i] = vec[i];
		}
		return  static_cast<bpn::array>( obj );
	}

}
