#ifndef PY_CPPAD_HPP
#define PY_CPPAD_HPP

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandle

#include "num_util.h"
#include "cppad/cppad.hpp"

using namespace std;
namespace b = boost;
namespace bp = boost::python;
namespace bpn = boost::python::numeric;
namespace nu = num_util;



namespace{
	typedef CppAD::AD<double> AD_double;
	typedef CppAD::AD<CppAD::AD<double> > ADD_double;

	class ADFun_double {
		private:
			CppAD::ADFun<double> f_;
		public:
			ADFun_double(bpn::array& x_array, bpn::array& y_array);
			bpn::array Forward(int p, bpn::array& xp);
	};

	class AD_double_vec {
		private:
			size_t       length_;
			AD_double  *pointer_;
			AD_double **handle_;
		public:
			typedef AD_double value_type;
			AD_double_vec(bpn::array& py_array);
			AD_double_vec(size_t length);
			AD_double_vec(const AD_double_vec& vec);
			AD_double_vec(void);
			~AD_double_vec(void);
			void operator=(const AD_double_vec& vec);
			size_t size(void) const;
			void resize(size_t length);
			AD_double& operator[](size_t i);
			const AD_double& operator[](size_t i) const;
	};

	class double_vec {
		private:
			size_t    length_;
			double  *pointer_;
			bool    allocated_;
		public:
			typedef double value_type;

			double_vec(bpn::array& py_array);
			double_vec(size_t length);
			double_vec(const double_vec& vec);
			double_vec(void);
			~double_vec(void);
			void operator=(const double_vec& vec);
			size_t size(void) const;
			void resize(size_t length);
			double& operator[](size_t i);
			const double& operator[](size_t i) const;
	};
	


	/* functions that make the wrapping a little more convenient */
	bpn::array vector2array(const double_vec& vec);

	/* general functions */
	void Independent(bpn::array& x_array);


	/* atomic (aka elementary) operations */
	AD_double	(*cos_AD_double) 		( const AD_double & ) = &CppAD::cos;
	AD_double	(*sin_AD_double) 		( const AD_double & ) = &CppAD::sin;
}

BOOST_PYTHON_MODULE(_cppad)
{
	using namespace boost::python;
	import_array(); 										/* some kind of hack to get numpy working */
	bpn::array::set_module_and_type("numpy", "ndarray");	/* some kind of hack to get numpy working */
	
	scope().attr("__doc__") =" CppAD: docstring \n\
							   next line";

	def("Independent", &Independent);


	class_<AD_double>("AD_double", init<double>())
		.def(boost::python::self_ns::str(self))
		
		.def(self+self)
		.def(self-self)
		.def(self*self)
		.def(self/self)

		.def(-self)
		.def(+self)
		.def(self += double() )
		.def(self -= double() )
		.def(self *= double() )
		.def(self /= double() )

		.def(self += self )
		.def(self -= self )
		.def(self *= self )
		.def(self /= self )

		.def("cos", cos_AD_double  )
		.def("sin", sin_AD_double  )
	;

	class_<ADFun_double>("ADFun_double", init< bpn::array& , bpn::array& >())
		.def("Forward", &ADFun_double::Forward)
	;
}

#endif
