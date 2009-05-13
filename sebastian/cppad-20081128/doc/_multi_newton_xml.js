var list_across0 = [
'_contents_xml.htm',
'_reference.xml',
'_index.xml',
'_search_xml.htm',
'_external.xml'
];
var list_up0 = [
'cppad.xml',
'adfun.xml',
'omp_max_thread.xml',
'openmp_run.sh.xml',
'multi_newton.cpp.xml',
'multi_newton.xml'
];
var list_down3 = [
'openmp_run.sh.xml'
];
var list_down2 = [
'example_a11c.cpp.xml',
'multi_newton.cpp.xml',
'sum_i_inv.cpp.xml'
];
var list_down1 = [
'multi_newton.xml',
'multi_newton.hpp.xml'
];
var list_current0 = [
'multi_newton.xml#Syntax',
'multi_newton.xml#Purpose',
'multi_newton.xml#Method',
'multi_newton.xml#xout',
'multi_newton.xml#fun',
'multi_newton.xml#n_grid',
'multi_newton.xml#xlow',
'multi_newton.xml#xup',
'multi_newton.xml#epsilon',
'multi_newton.xml#max_itr'
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
