var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'appendix.htm',
'speed.htm',
'speed_utility.htm',
'sparse_evaluate.htm'
];
var list_down3 = [
'faq.htm',
'speed.htm',
'theory.htm',
'glossary.htm',
'bib.htm',
'bugs.htm',
'wishlist.htm',
'whats_new.htm',
'include_deprecated.htm',
'license.htm'
];
var list_down2 = [
'speed_main.htm',
'speed_utility.htm',
'speed_double.htm',
'speed_adolc.htm',
'speed_cppad.htm',
'speed_fadbad.htm',
'speed_sacado.htm'
];
var list_down1 = [
'uniform_01.htm',
'det_of_minor.htm',
'det_by_minor.htm',
'det_by_lu.htm',
'det_33.htm',
'det_grad_33.htm',
'ode_evaluate.htm',
'sparse_evaluate.htm'
];
var list_down0 = [
'sparse_evaluate.cpp.htm',
'sparse_evaluate.hpp.htm'
];
var list_current0 = [
'sparse_evaluate.htm#Syntax',
'sparse_evaluate.htm#Purpose',
'sparse_evaluate.htm#Inclusion',
'sparse_evaluate.htm#Float',
'sparse_evaluate.htm#x',
'sparse_evaluate.htm#i',
'sparse_evaluate.htm#j',
'sparse_evaluate.htm#m',
'sparse_evaluate.htm#fm',
'sparse_evaluate.htm#fm.Function',
'sparse_evaluate.htm#fm.Gradient',
'sparse_evaluate.htm#fm.Hessian',
'sparse_evaluate.htm#Example',
'sparse_evaluate.htm#Source Code'
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
