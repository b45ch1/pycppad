#include "py_cppad.hpp"


/* operators */
AD_double *AD_double_mul_AD_double_AD_double(const AD_double &lhs, const AD_double &rhs){
	return new AD_double(CppAD::operator*(lhs,rhs));
}


/* functions */
void Independent_numpy_AD_double(bpn::array &bpn_x){
	/* input checks of the numpy array */
	if(!nu::iscontiguous(bpn_x)){
		printf("not a contiguous array!\n");
	}
	nu::check_rank(bpn_x,1);

	vector<intp> shp(nu::shape(bpn_x));
	int N = shp[0];
	
	CppAD::vector<AD_double> x(N);
	bp::object* obj_x = (bp::object*) nu::data(bpn_x);
	for(int n=0; n!=N; ++n){
		x[n] = bp::extract<AD_double&>(obj_x[n])();
	}
	CppAD::Independent(x);
}
