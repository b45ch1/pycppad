var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'adfun.htm',
'omp_max_thread.htm',
'openmp_run.sh.htm',
'multi_newton.cpp.htm',
'multi_newton.htm'
];
var list_down3 = [
'openmp_run.sh.htm'
];
var list_down2 = [
'example_a11c.cpp.htm',
'multi_newton.cpp.htm',
'sum_i_inv.cpp.htm'
];
var list_down1 = [
'multi_newton.htm',
'multi_newton.hpp.htm'
];
var list_current0 = [
'multi_newton.htm#Syntax',
'multi_newton.htm#Purpose',
'multi_newton.htm#Method',
'multi_newton.htm#xout',
'multi_newton.htm#fun',
'multi_newton.htm#n_grid',
'multi_newton.htm#xlow',
'multi_newton.htm#xup',
'multi_newton.htm#epsilon',
'multi_newton.htm#max_itr'
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
