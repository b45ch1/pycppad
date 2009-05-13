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
'speed_main.htm'
];
var list_down3 = [
'install.htm',
'introduction.htm',
'ad.htm',
'adfun.htm',
'library.htm',
'preprocessor.htm',
'example.htm',
'appendix.htm'
];
var list_down2 = [
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
var list_down1 = [
'speed_main.htm',
'speed_utility.htm',
'speed_double.htm',
'speed_adolc.htm',
'speed_cppad.htm',
'speed_fadbad.htm',
'speed_sacado.htm'
];
var list_down0 = [
'link_det_lu.htm',
'link_det_minor.htm',
'link_poly.htm',
'link_sparse_hessian.htm',
'link_ode.htm'
];
var list_current0 = [
'speed_main.htm#Syntax',
'speed_main.htm#Purpose',
'speed_main.htm#package',
'speed_main.htm#package.AD Package',
'speed_main.htm#package.double',
'speed_main.htm#package.profile',
'speed_main.htm#option',
'speed_main.htm#option.correct',
'speed_main.htm#option.speed',
'speed_main.htm#seed',
'speed_main.htm#retape',
'speed_main.htm#Correctness Results',
'speed_main.htm#Speed Results',
'speed_main.htm#Contents'
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
