# ifndef PYCPPAD_VEC2ARRAY_INCLUDED
# define PYCPPAD_VEC2ARRAY_INCLUDED

# include "environment.hpp"
# include "vector.hpp"

namespace pycppad {
	array vec2array(double_vec& vec);
	array vec2array(AD_double_vec& vec);
	array vec2array(AD_AD_double_vec& vec);

	// some kind of hack connected to numeric::array
	void vec2array_import_array(void);
}

# endif
