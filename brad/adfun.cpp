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
	array ADFun_AD_double::Forward(int p, array& xp)
	{	size_t        p_sz(p);
		AD_double_vec xp_vec(xp);
		AD_double_vec result = f_.Forward(p_sz, xp_vec);

		array retvalue = vector2array(result);
		return retvalue;
	}
	void adfun_avoid_warning_that_import_array_not_used(void)
	{	import_array(); }
}
