# include "vec2array.hpp"

# define PY_ARRAY_UNIQUE_SYMBOL PyArray_Pycppad

# define PYCPPAD_DEBUG_ISSUES

namespace pycppad {
array vec2array(double_vec& vec)
{	int n = static_cast<int>( vec.size() );
	PYCPPAD_ASSERT( n >= 0 , "");

	object obj(handle<>( PyArray_FromDims(1, &n, PyArray_DOUBLE) ));
	double *ptr = static_cast<double*> ( PyArray_DATA (
		reinterpret_cast<PyArrayObject*> ( obj.ptr() )
	));
	for(size_t i = 0; i < vec.size(); i++){
		ptr[i] = vec[i];
	}
	return  static_cast<array>( obj );
}
array vec2array(AD_double_vec& vec)
{
	int n = static_cast<int>( vec.size() );
	PYCPPAD_ASSERT( n >= 0 , "");

	object obj(handle<>( PyArray_FromDims(1, &n, PyArray_OBJECT) ));
	for(size_t i = 0; i < vec.size(); i++){
		obj[i] = vec[i];
	}
	return  static_cast<array>( obj );
}
array vec2array(AD_AD_double_vec& vec)
{	int n = static_cast<int>( vec.size() );
	PYCPPAD_ASSERT( n >= 0 , "");

	object obj(handle<>( PyArray_FromDims(1, &n, PyArray_OBJECT) ));
	for(size_t i = 0; i < vec.size(); i++){
		obj[i] = vec[i];
	}
	return  static_cast<array>( obj );
}
array vec2array(size_t m, size_t n, double_vec& vec)
{
	PYCPPAD_ASSERT(m * n == vec.size(), "");

	int dims[2];
	dims[0] = static_cast<int>(m);
	dims[1] = static_cast<int>(n);
	PYCPPAD_ASSERT( dims[0] >= 0, "");
	PYCPPAD_ASSERT( dims[1] >= 0, "");
	object obj(handle<>( 
		PyArray_FromDims(2, dims, PyArray_DOUBLE) 
	));
	double *ptr = static_cast<double*> ( PyArray_DATA (
		reinterpret_cast<PyArrayObject*> ( obj.ptr() )
	));
	size_t i = vec.size();
	while(i--)
		ptr[i] = vec[i];
	return  static_cast<array>( obj );
}
array vec2array(size_t m, size_t n, AD_double_vec& vec)
{
	PYCPPAD_ASSERT(m * n == vec.size(), "");

	int dims[2];
	dims[0] = static_cast<int>(m);
	dims[1] = static_cast<int>(n);
	PYCPPAD_ASSERT( dims[0] >= 0, "");
	PYCPPAD_ASSERT( dims[1] >= 0, "");
	object obj(handle<>( 
		PyArray_FromDims(2, dims, PyArray_OBJECT) 
	));
	size_t i = vec.size();
	for(i = 0; i < vec.size(); i++)
	{
# ifdef PYCPPAD_DEBUG_ISSUES
std::cout << "vec2array: i = " << i << std::endl;
# endif
		obj[i] = vec[i];
	}
	return  static_cast<array>( obj );
}
// ========================================================================
void vec2array_import_array(void)
{	import_array(); }
} // end namespace pycppad
