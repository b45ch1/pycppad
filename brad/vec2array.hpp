# ifndef PYCPPAD_VEC2ARRAY_INCLUDED
# define PYCPPAD_VEC2ARRAY_INCLUDED

# include "setup.hpp"
# include "vector.hpp"

namespace pycppad {
	array vec2array(const double_vec& vec);
	array vec2array(const AD_double_vec& vec);
	array vec2array(const AD_AD_double_vec& vec);
	array vec2array(size_t m, size_t n, const double_vec& vec);
	array vec2array(size_t m, size_t n, const AD_double_vec& vec);

	// some kind of hack connected to numeric::array
	void vec2array_import_array(void);
}

# endif
