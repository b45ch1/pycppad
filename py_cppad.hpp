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

/* Base types */
typedef CppAD::AD<double> AD_double;
typedef CppAD::AD<CppAD::AD<double> > ADD_double;

/* vector types */
typedef CppAD::vector<double> vector_double;
typedef CppAD::vector<CppAD::AD<double> > vector_AD_double;
typedef CppAD::vector<CppAD::AD<CppAD::AD<double> > > vector_ADD_double;

/* ??? types */
typedef CppAD::VecAD<double> VecAD_double;
typedef CppAD::VecAD< CppAD::AD<double> > VecAD_AD_double;

/* ADFun types */
typedef CppAD::ADFun<double> ADFun_double;
typedef CppAD::ADFun<CppAD::AD<double> > ADFun_AD_double;
typedef CppAD::ADFun<CppAD::AD<CppAD::AD<double> > > ADFun_ADD_double;


/* elementary binary operators */
AD_double *AD_double_mul_AD_double_AD_double(const AD_double &lhs, const AD_double &rhs);

/* elementary unary operators */
AD_double	(*cos_AD_double) 		( const AD_double & ) = &CppAD::cos;

/* general functions */
void Independent_numpy_AD_double(bpn::array &bpn_x);
AD_double *AD_double_mul_AD_double_AD_double(const AD_double &lhs, const AD_double &rhs);

BOOST_PYTHON_MODULE(_cppad)
{
	using namespace boost::python;
	import_array(); 										/* some kind of hack to get numpy working */
	bpn::array::set_module_and_type("numpy", "ndarray");	/* some kind of hack to get numpy working */
	
	scope().attr("__doc__") =" CppAD: docstring \n\
							   next line";

	def("Independent",		Independent_numpy_AD_double);


	class_<AD_double>("AD_double", init<double>())
		.def(boost::python::self_ns::str(self))
		.def("__mul__", AD_double_mul_AD_double_AD_double, return_value_policy<manage_new_object>())
		.def("cos", cos_AD_double  )

	;
// 	class_<ADFun_double>("ADFun_double")
// 	.def(init<const bpn::array&,const bpn::array&>())
// 	;
}

#endif
