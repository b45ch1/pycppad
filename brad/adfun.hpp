# ifndef PYCPPAD_ADFUN_INCLUDED
# define PYCPPAD_ADFUN_INCLUDED

# include "setup.hpp"

namespace pycppad {
	// -------------------------------------------------------------
	// class ADFun<Base>
	template <class Base>
	class ADFun{
	private:
		CppAD::ADFun<Base> f_;
	public:
		// constructor
		ADFun(array& x_array, array& y_array);
		// member functions
		int   Domain(void);
		int   Range(void);
		array Forward(int p, array& xp);
		array Reverse(int p, array& w);
		array Jacobian(array& x);
		array Hessian(array& x, array& w);
	};
	typedef ADFun<double>    ADFun_double;
	typedef ADFun<AD_double> ADFun_AD_double;
}

# endif
