# include "adfun.hpp"
# include "vector.hpp"

namespace python_cppad {
	// -------------------------------------------------------------
	// class ADFun_double
	//
	ADFun_double::ADFun_double(array& x_array, array& y_array)
	{	AD_double_vec x_vec(x_array);
		AD_double_vec y_vec(y_array);

		f_.Dependent(x_vec, y_vec);
	}
	// member function Forward
	array ADFun_double::Forward(int p, array& xp)
	{	size_t     p_sz(p);
		double_vec xp_vec(xp);
		double_vec result = f_.Forward(p_sz, xp_vec);
		return vector2array(result);
	}
	// -------------------------------------------------------------
	// class ADFun_AD_double
	//
	ADFun_AD_double::ADFun_AD_double(array& x_array, array& y_array)
	{	AD_AD_double_vec x_vec(x_array);
		AD_AD_double_vec y_vec(y_array);

		f_.Dependent(x_vec, y_vec);
	}
	// Kludge: Version of Forward member function until we figure
	// out how to retrun an array with AD_double elements.
	// (This should look like forward for ADFun_double with
	// double replaced by AD_double.)
	void ADFun_AD_double::Forward(int p, array& xp, array& fp)
	{	size_t        p_sz(p);
		AD_double_vec xp_vec(xp);
		AD_double_vec result = f_.Forward(p_sz, xp_vec);

		// kludge to pass back result
		int* dims_ptr = PyArray_DIMS(fp.ptr());
		int ndim      = PyArray_NDIM(fp.ptr());
		size_t length = static_cast<size_t>( dims_ptr[0] );
		PYTHON_CPPAD_ASSERT( ndim == 1 , 
			"forward: third argument is not a vector."
		);
		PYTHON_CPPAD_ASSERT( length == f_.Range() , 
			"foward: third argument length not equal "
			"range dimension for function."
		);
		object *obj_ptr = static_cast<object*>( 
			PyArray_DATA(fp.ptr()) 
		);
		for(size_t i = 0; i < result.size(); i++)
		{	AD_double *ptr = 	
				& extract<AD_double&>(obj_ptr[i])(); 
			*ptr = result[i];
		}
		return;
	}
	void adfun_avoid_warning_that_import_array_not_used(void)
	{	import_array(); }
}
