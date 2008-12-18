# include "vector.hpp"

namespace python_cppad {
// ========================================================================
// class double_vec
//
// constructor from a python array
double_vec::double_vec(array& py_array)
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
// ---------------------------------------------------------------------------
// class vec<Scalar>
//
template <class Scalar>
vec<Scalar>::vec(array& py_array)
{
	// get array info
	int* dims_ptr = PyArray_DIMS(py_array.ptr());
	int ndim      = PyArray_NDIM(py_array.ptr());
	int length    = dims_ptr[0];

	// check array info
	PYTHON_CPPAD_ASSERT( ndim == 1 , "Argument is not a vector.");
	PYTHON_CPPAD_ASSERT( length >= 0 , "");
	// pointer to object
	object *obj_ptr = static_cast<object*>( 
		PyArray_DATA(py_array.ptr()) 
	);

	// set private data
	length_  = static_cast<size_t>(length);
	pointer_ = 0;
	handle_  = CPPAD_TRACK_NEW_VEC(length_, handle_);
	for(size_t i = 0; i < length_; i++) handle_[i] = 
		& extract<Scalar&>(obj_ptr[i])(); 
	return;
}
// constructor from size
template <class Scalar>
vec<Scalar>::vec(size_t length)
{
	length_  = length;
	pointer_ = CPPAD_TRACK_NEW_VEC(length, pointer_);
	handle_  = CPPAD_TRACK_NEW_VEC(length, handle_);
	for(size_t i = 0; i < length_; i++)
		handle_[i] = pointer_ + i;
	return;
}

// copy constructor
template <class Scalar>
vec<Scalar>::vec(const vec& vec)
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
template <class Scalar>
vec<Scalar>::vec(void)
{
	length_  = 0;
	pointer_ = 0;
	handle_  = 0;
}

// destructor
template <class Scalar>
vec<Scalar>::~vec(void)
{
	if( handle_ != 0 )
		CPPAD_TRACK_DEL_VEC(handle_); 
	if( pointer_ != 0 )
		CPPAD_TRACK_DEL_VEC(pointer_);	
}

// assignment operator
template <class Scalar>
void vec<Scalar>::operator=(const vec& vec)
{
	PYTHON_CPPAD_ASSERT( length_ == vec.length_ , ""); 
	for(size_t i = 0; i < length_; i++)
		*handle_[i] = *(vec.handle_[i]);
	return;
}

// size member function
template <class Scalar>
size_t vec<Scalar>::size(void) const
{	return length_; }

// resize 
template <class Scalar>
void vec<Scalar>::resize(size_t length)
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
template <class Scalar>
Scalar& vec<Scalar>::operator[](size_t i)
{	assert( i < length_ );
	return *handle_[i];
}

// constant element access
template <class Scalar>
const Scalar& vec<Scalar>::operator[](size_t i) const
{	assert( i < length_ );
	return *handle_[i];
}

// instantiate instances of template class
template class vec<AD_double>;
template class vec<AD_AD_double>;
// ========================================================================
} // end namespace python_cppad
