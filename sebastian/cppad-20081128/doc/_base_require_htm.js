var list_across0 = [
'_contents.htm',
'_reference.htm',
'_index.htm',
'_search.htm',
'_external.htm'
];
var list_up0 = [
'cppad.htm',
'ad.htm',
'base_require.htm'
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
'default.htm',
'ad_copy.htm',
'convert.htm',
'advalued.htm',
'boolvalued.htm',
'vecad.htm',
'base_require.htm'
];
var list_down0 = [
'base_complex.hpp.htm',
'base_adolc.hpp.htm'
];
var list_current0 = [
'base_require.htm#Purpose',
'base_require.htm#Warning',
'base_require.htm#Numeric Type',
'base_require.htm#declare.hpp',
'base_require.htm#CondExp',
'base_require.htm#CondExp.Ordered Type',
'base_require.htm#CondExp.Not Ordered',
'base_require.htm#EqualOpSeq',
'base_require.htm#EqualOpSeq.Suggestion',
'base_require.htm#Identical',
'base_require.htm#Identical.Suggestion',
'base_require.htm#Integer',
'base_require.htm#Integer.Suggestion',
'base_require.htm#Ordered',
'base_require.htm#Ordered.Ordered Type',
'base_require.htm#Ordered.Not Ordered',
'base_require.htm#pow',
'base_require.htm#Standard Math Unary',
'base_require.htm#Example'
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
