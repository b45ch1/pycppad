var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'example.htm',
'general.htm',
'ipopt_cppad_nlp.htm',
'ipopt_cppad_ode.cpp.htm'
];
var list_down3 = [
'general.htm',
'exampleutility.htm',
'listallexamples.htm',
'test_vector.htm'
];
var list_down2 = [
'ipopt_cppad_nlp.htm',
'interface2c.cpp.htm',
'jacminordet.cpp.htm',
'jacludet.cpp.htm',
'hesminordet.cpp.htm',
'hesludet.cpp.htm',
'odestiff.cpp.htm',
'ode_taylor.cpp.htm',
'ode_taylor_adolc.cpp.htm',
'stackmachine.cpp.htm',
'mul_level.htm'
];
var list_down1 = [
'ipopt_cppad_windows.htm',
'ipopt_cppad_simple.cpp.htm',
'ipopt_cppad_ode.cpp.htm'
];
var list_current0 = [
'ipopt_cppad_ode.cpp.htm#Purpose',
'ipopt_cppad_ode.cpp.htm#General Problem',
'ipopt_cppad_ode.cpp.htm#ODE Discrete Approximation',
'ipopt_cppad_ode.cpp.htm#Optimization Problem',
'ipopt_cppad_ode.cpp.htm#eval_r(k, u)',
'ipopt_cppad_ode.cpp.htm#Source Code'
];
function choose_across0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_across0[index-1];
}
function choose_up0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_up0[index-1];
}
function choose_down3(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down3[index-1];
}
function choose_down2(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down2[index-1];
}
function choose_down1(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down1[index-1];
}
function choose_down0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_down0[index-1];
}
function choose_current0(item)
{	var index          = item.selectedIndex;
	item.selectedIndex = 0;
	if(index > 0)
		document.location = list_current0[index-1];
}
