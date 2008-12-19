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
		// member function Forward
		array Forward(int p, array& xp);
	};
	typedef ADFun<double>    ADFun_double;
	typedef ADFun<AD_double> ADFun_AD_double;
}

# endif
