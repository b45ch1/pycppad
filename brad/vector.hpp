# ifndef PYTHON_CPPAD_VECTOR_INCLUDED
# define PYTHON_CPPAD_VECTOR_INCLUDED

# include "setup.hpp"

namespace python_cppad {
// ===================================================================
class double_vec {
private:
	size_t    length_;  // set by constructor only
	double  *pointer_;  // set by constructor only
	bool    allocated_; // set by constructor only
public:
	typedef double value_type;

	// constructor from a python array
	double_vec(array& py_array);

	// constructor from size
	double_vec(size_t length);

	// copy constructor
	double_vec(const double_vec& vec);

	// default constructor
	double_vec(void);

	// destructor
	~double_vec(void);

	// assignment operator
	void operator=(const double_vec& vec);

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
class AD_double_vec {
private:
	size_t       length_; // set by constructor only
	AD_double  *pointer_; // set by constructor only
	AD_double **handle_;  // set by constructor only
public:
	typedef AD_double value_type;

	// constructor from a python array
	AD_double_vec(array& py_array);

	// constructor from size
	AD_double_vec(size_t length);

	// copy constructor
	AD_double_vec(const AD_double_vec& vec);

	// default constructor
	AD_double_vec(void);

	// destructor
	~AD_double_vec(void);

	// assignment operator
	void operator=(const AD_double_vec& vec);

	// size member function
	size_t size(void) const;

	// resize member function
	void resize(size_t length);

	// non constant element access
	AD_double& operator[](size_t i);

	// constant element access
	const AD_double& operator[](size_t i) const;
};
// ------------------------------------------------------------------------
class AD_AD_double_vec {
private:
	size_t       length_; // set by constructor only
	AD_AD_double  *pointer_; // set by constructor only
	AD_AD_double **handle_;  // set by constructor only
public:
	typedef AD_AD_double value_type;

	// constructor from a python array
	AD_AD_double_vec(array& py_array);

	// constructor from size
	AD_AD_double_vec(size_t length);

	// copy constructor
	AD_AD_double_vec(const AD_AD_double_vec& vec);

	// default constructor
	AD_AD_double_vec(void);

	// destructor
	~AD_AD_double_vec(void);

	// assignment operator
	void operator=(const AD_AD_double_vec& vec);

	// size member function
	size_t size(void) const;

	// resize member function
	void resize(size_t length);

	// non constant element access
	AD_AD_double& operator[](size_t i);

	// constant element access
	const AD_AD_double& operator[](size_t i) const;
};
// ========================================================================
} // end of python_cppad namespace

# endif
