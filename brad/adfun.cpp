# include "adfun.hpp"
# include "vector.hpp"

namespace pycppad {
	// -------------------------------------------------------------
	// class ADFun<Base>
	template <class Base>
	ADFun<Base>::ADFun(array& x_array, array& y_array)
	{	vec< CppAD::AD<Base> > x_vec(x_array);
		vec< CppAD::AD<Base> > y_vec(y_array);

		f_.Dependent(x_vec, y_vec);
	}
	// member function Forward
	template <class Base>
	array ADFun<Base>::Forward(int p, array& xp)
	{	size_t    p_sz(p);
		vec<Base> xp_vec(xp);
		vec<Base> result = f_.Forward(p_sz, xp_vec);
		return vector2array(result);
	}
	// -------------------------------------------------------------
	// instantiate instances of ADFun<Base>
	template class ADFun<double>;
	template class ADFun<AD_double>;
	// -------------------------------------------------------------
	void adfun_avoid_warning_that_import_array_not_used(void)
	{	import_array(); }
}
