# ifndef PYCPPAD_VECTOR_INCLUDED
# define PYCPPAD_VECTOR_INCLUDED

# include "setup.hpp"

namespace pycppad {
// ------------------------------------------------------------------------
template <class Scalar>
class vec {
private:
	size_t    length_; // set by constructor only
	Scalar  *pointer_; // set by constructor only
	Scalar  **handle_; // set by constructor only
public:
	typedef Scalar value_type;

	// constructor from a python array
	vec(array& py_array);

	// constructor from size
	vec(size_t length);

	// copy constructor
	vec(const vec& vec);

	// default constructor
	vec(void);

	// destructor
	~vec(void);

	// assignment operator
	void operator=(const vec& vec);

	// size member function
	size_t size(void) const;

	// resize member function
	void resize(size_t length);

	// non constant element access
	Scalar& operator[](size_t i);

	// constant element access
	const Scalar& operator[](size_t i) const;
};
// ===================================================================
template <>
class vec<double> {
private:
	size_t    length_;  // set by constructor only
	double  *pointer_;  // set by constructor only
	bool    allocated_; // set by constructor only
public:
	typedef double value_type;

	// constructor from a python array
	vec(array& py_array);

	// constructor from size
	vec(size_t length);

	// copy constructor
	vec(const vec& vec);

	// default constructor
	vec(void);

	// destructor
	~vec(void);

	// assignment operator
	void operator=(const vec& vec);

	// size member function
	size_t size(void) const;

	// resize member function
	void resize(size_t length);

	// non constant element access
	double& operator[](size_t i);

	// constant element access
	const double& operator[](size_t i) const;
};
// ------------------------------------------------------------------------
typedef vec<double>       double_vec;
typedef vec<AD_double>    AD_double_vec;
typedef vec<AD_AD_double> AD_AD_double_vec;

array vector2array(const double_vec& vec);
array vector2array(const AD_double_vec& vec);

array vector2array(size_t m, size_t n, const double_vec& vec);
array vector2array(size_t m, size_t n, const AD_double_vec& vec);
// ========================================================================
} // end of pycppad namespace

# endif
