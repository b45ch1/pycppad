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
'wishlist.htm'
];
var list_down2 = [
'install.htm',
'introduction.htm',
'ad.htm',
'adfun.htm',
'library.htm',
'preprocessor.htm',
'example.htm',
'appendix.htm'
];
var list_down1 = [
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
var list_current0 = [
'wishlist.htm#Atan2',
'wishlist.htm#BenderQuad',
'wishlist.htm#CondExp',
'wishlist.htm#Exceptions',
'wishlist.htm#Ipopt',
'wishlist.htm#Library',
'wishlist.htm#Multiple Arguments',
'wishlist.htm#Numeric Limits',
'wishlist.htm#Operation Sequence',
'wishlist.htm#Optimization',
'wishlist.htm#Optimization.Expression Hashing',
'wishlist.htm#Optimization.Microsoft Compiler',
'wishlist.htm#Optimization.Remove Operations From Tape',
'wishlist.htm#Scripting Languages',
'wishlist.htm#Software Guidelines',
'wishlist.htm#Software Guidelines.Boost',
'wishlist.htm#Sparse Jacobians and Hessians',
'wishlist.htm#Sparsity Patterns',
'wishlist.htm#Speed Testing',
'wishlist.htm#Tan and Tanh',
'wishlist.htm#Tracing',
'wishlist.htm#VecAD',
'wishlist.htm#Vector Element Type'
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
