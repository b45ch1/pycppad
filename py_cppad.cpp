#include "py_cppad.hpp"



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

	/* create new tape */
	size_t id = AD_double::tape_new();
// 	AD_double::tape_ptr(id)->Independent_numpy_AD_double(bpn_x);

		
// 	vector<intp> shp(nu::shape(bpn_x));
// 	int N = shp[0];
// 	CppAD::vector<AD_double> x(N);
// 	bp::object* in_x = (bp::object*) nu::data(bpn_x);
// 
// 	cout<<"type="<<nu::type(bpn_x)<<endl;
// Vec2& v = extract<Vec2&>(o);
// 	for(int n=0; n!=N; ++n){
// 		x[n] = bp::extract<AD_double&>(in_x[n]);
// 	}
// // 	CppAD::vector<AD_double> x((AD_double*) nu::data(bpn_x), N);
// 	cout<<x[0]<<endl;

}


// void AD_double::Independent_numpy_AD_double(bpn::array &bpn_x)
// {
// 	bp::object* in_x = (bp::object*) nu::data(bpn_x);
// 	vector<intp> shp(nu::shape(bpn_x));
// 	size_t n = shp[0];
// 
// 	CPPAD_ASSERT_UNKNOWN( Rec_.TotNumVar() == 0 );
// 
// 	// skip the first record (parameters use taddr zero)
// 	CPPAD_ASSERT_UNKNOWN( NumVar(NonOp) == 1 );
// 	CPPAD_ASSERT_UNKNOWN( NumInd(NonOp) == 0 );
// 	Rec_.PutOp(NonOp);
// 
// 
// 	// place each of the independent variables in the tape
// 	CPPAD_ASSERT_UNKNOWN( NumVar(InvOp) == 1 );
// 	CPPAD_ASSERT_UNKNOWN( NumInd(InvOp) == 0 );
// 	size_t j;
// 	for(j = 0; j < n; j++)
// 	{	// tape address for this independent variable
// 		x[j].taddr_ = Rec_.PutOp(InvOp);
// 		x[j].id_    = id_;
// 		CPPAD_ASSERT_UNKNOWN( bp::extract<AD_double&>(in_x[j]).taddr_ == j+1 );
// 		CPPAD_ASSERT_UNKNOWN( Variable(bp::extract<AD_double&>(in_x[j]) ) );
// 	}
// 
// 	// done specifying all of the independent variables
// 	size_independent_ = n;
// }

// template <typename VectorAD>
// inline void Independent(VectorAD &x)
// {	typedef typename VectorAD::value_type ADBase;
// 	typedef typename ADBase::value_type   Base;
// 	CPPAD_ASSERT_KNOWN(
// 		ADBase::tape_ptr() == CPPAD_NULL,
// 		"Independent: cannot create a new tape because"
// 		"\na previous tape is still active (for this thread)."
// 	);
// 	size_t id = ADBase::tape_new();
// 
// 	ADBase::tape_ptr(id)->Independent(x); 
// }
// 
