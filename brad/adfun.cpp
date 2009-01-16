# include "adfun.hpp"
# include "vector.hpp"
# include "vec2array.hpp"

namespace pycppad {
	// -------------------------------------------------------------
	// class ADFun<Base>
	template <class Base>
	ADFun<Base>::ADFun(array& x_array, array& y_array)
	{	vec< CppAD::AD<Base> > x_vec(x_array);
		vec< CppAD::AD<Base> > y_vec(y_array);

		f_.Dependent(x_vec, y_vec);
	}
	// member functions Domain and Range
	template <class Base>
	int ADFun<Base>::Domain(void)
	{	return static_cast<int>( f_.Domain() ); }
	template <class Base>
	int ADFun<Base>::Range(void)
	{	return static_cast<int>( f_.Range() ); }
	// member function Forward
	template <class Base>
	array ADFun<Base>::Forward(int p, array& xp)
	{	size_t    p_sz(p);
		vec<Base> xp_vec(xp);
		vec<Base> result = f_.Forward(p_sz, xp_vec);
		return vec2array(result);
	}
	// member function Reverse
	template <class Base>
	array ADFun<Base>::Reverse(int p, array& w)
	{	size_t    p_sz(p);
		vec<Base> w_vec(w);
		vec<Base> dw_vec = f_.Reverse(p_sz, w_vec);
		size_t n = f_.Domain();
		vec<Base> result(n);
		for(size_t j = 0; j < n; j++)
			result[j] = dw_vec[j*p + p - 1];
		return vec2array(result);
	}
	// member function Jacobian
	template <class Base>
	array ADFun<Base>::Jacobian(array& x)
	{	vec<Base> x_vec(x);
		vec<Base> result = f_.Jacobian(x_vec);
		return vec2array(result);
	}
	// -------------------------------------------------------------
	// instantiate instances of ADFun<Base>
	template class ADFun<double>;
	template class ADFun<AD_double>;
	// -------------------------------------------------------------
	void adfun_avoid_warning_that_import_array_not_used(void)
	{	import_array(); }
}
